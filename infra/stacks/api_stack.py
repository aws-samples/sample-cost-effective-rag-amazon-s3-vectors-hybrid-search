"""API stack: REST API with request validation, API key, usage plan, throttling.

Endpoints (both require ``x-api-key``):
    POST /search  -> hybrid retrieval only (no LLM call)
    POST /query   -> hybrid retrieval + grounded generation with citations

Responses are buffered JSON. REST APIs support the STREAM response transfer
mode (proxy integrations) since 2025-11-19, but the Python managed runtime
can't stream a response; streaming needs a Node.js or Lambda Web Adapter
front (Function URL with RESPONSE_STREAM, or a REST method in STREAM mode).
See docs/architecture.md.

The ``prod`` stage writes JSON access logs (including
``$context.identity.apiKeyId``) to the ApiAccessLogs log group (14-day
retention). To deliver them, the stack sets the account-level API Gateway
CloudWatch Logs role (``cloudWatchRoleArn``). There is one such role per
account per Region, shared by all REST APIs, so the role is retained when
the stack is destroyed.
"""
import json

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_lambda as _lambda,
    aws_logs as logs,
)
from constructs import Construct

from .config import PatternConfig

# JSON Schema pattern for ``tenant_id`` on /search and /query. The handlers
# trim and lowercase the value (src/common/tenant.py), and blank means the
# default tenant, so the edge accepts exactly what normalizes to a valid
# ``^[a-z0-9][a-z0-9_-]{0,63}$`` ID: optional surrounding whitespace around
# an empty string or a 1-64 character ID in either letter case. Keep this in
# sync with src/common/tenant.py.
TENANT_ID_SCHEMA_PATTERN = r"^\s*([A-Za-z0-9][A-Za-z0-9_-]{0,63})?\s*$"


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: PatternConfig,
        search_fn: _lambda.IFunction,
        query_fn: _lambda.IFunction,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        access_logs = logs.LogGroup(
            self,
            "ApiAccessLogs",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        access_log_format = apigw.AccessLogFormat.custom(
            json.dumps(
                {
                    "requestId": "$context.requestId",
                    "ip": "$context.identity.sourceIp",
                    "caller": "$context.identity.caller",
                    "user": "$context.identity.user",
                    "apiKeyId": "$context.identity.apiKeyId",
                    "requestTime": "$context.requestTime",
                    "httpMethod": "$context.httpMethod",
                    "resourcePath": "$context.resourcePath",
                    "status": "$context.status",
                    "protocol": "$context.protocol",
                    "responseLength": "$context.responseLength",
                }
            )
        )

        api = apigw.RestApi(
            self,
            "Api",
            rest_api_name=f"{cfg.prefix}-api",
            description="Hybrid RAG on Amazon S3 Vectors",
            # Account-level role (one per account per Region, shared by all
            # REST APIs) that lets API Gateway write the access logs; retained
            # on destroy so other APIs in the account keep logging.
            cloud_watch_role=True,
            cloud_watch_role_removal_policy=RemovalPolicy.RETAIN,
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=cfg.api_throttle_rate,
                throttling_burst_limit=cfg.api_throttle_burst,
                metrics_enabled=True,
                access_log_destination=apigw.LogGroupLogDestination(access_logs),
                access_log_format=access_log_format,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "X-Api-Key"],
            ),
        )

        validator = api.add_request_validator(
            "BodyValidator", validate_request_body=True, validate_request_parameters=False
        )

        search_model = api.add_model(
            "SearchModel",
            content_type="application/json",
            schema=apigw.JsonSchema(
                schema=apigw.JsonSchemaVersion.DRAFT4,
                title="SearchRequest",
                type=apigw.JsonSchemaType.OBJECT,
                required=["query"],
                properties={
                    "query": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING, min_length=1),
                    "tenant_id": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.STRING, pattern=TENANT_ID_SCHEMA_PATTERN
                    ),
                    "top_k": apigw.JsonSchema(type=apigw.JsonSchemaType.INTEGER, minimum=1, maximum=50),
                    "mode": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.STRING, enum=["hybrid", "dense", "sparse"]
                    ),
                },
            ),
        )

        query_model = api.add_model(
            "QueryModel",
            content_type="application/json",
            schema=apigw.JsonSchema(
                schema=apigw.JsonSchemaVersion.DRAFT4,
                title="QueryRequest",
                type=apigw.JsonSchemaType.OBJECT,
                required=["question"],
                properties={
                    "question": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING, min_length=1),
                    "tenant_id": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.STRING, pattern=TENANT_ID_SCHEMA_PATTERN
                    ),
                    "top_k": apigw.JsonSchema(type=apigw.JsonSchemaType.INTEGER, minimum=1, maximum=20),
                    "max_tokens": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.INTEGER, minimum=1, maximum=4096
                    ),
                    "temperature": apigw.JsonSchema(
                        type=apigw.JsonSchemaType.NUMBER, minimum=0, maximum=1
                    ),
                },
            ),
        )

        api.root.add_resource("search").add_method(
            "POST",
            apigw.LambdaIntegration(search_fn),
            api_key_required=True,
            request_validator=validator,
            request_models={"application/json": search_model},
        )
        api.root.add_resource("query").add_method(
            "POST",
            apigw.LambdaIntegration(query_fn),
            api_key_required=True,
            request_validator=validator,
            request_models={"application/json": query_model},
        )

        key = api.add_api_key("ApiKey", api_key_name=f"{cfg.prefix}-key")
        plan = api.add_usage_plan(
            "UsagePlan",
            name=f"{cfg.prefix}-plan",
            throttle=apigw.ThrottleSettings(
                rate_limit=cfg.api_throttle_rate, burst_limit=cfg.api_throttle_burst
            ),
            quota=apigw.QuotaSettings(limit=cfg.api_monthly_quota, period=apigw.Period.MONTH),
        )
        plan.add_api_stage(stage=api.deployment_stage)
        plan.add_api_key(key)

        CfnOutput(self, "ApiEndpoint", value=api.url)
        CfnOutput(
            self,
            "ApiKeyId",
            value=key.key_id,
            description="Retrieve the value with: aws apigateway get-api-key "
            "--api-key <id> --include-value --query value --output text",
        )
