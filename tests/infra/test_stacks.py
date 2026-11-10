"""Synth-level assertions on the CDK stacks (asset bundling skipped)."""
import json
import re

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from common.tenant import resolve_request_tenant
from ingestion.parsers import SUPPORTED_FORMATS
from stacks.api_stack import TENANT_ID_SCHEMA_PATTERN, ApiStack
from stacks.config import PatternConfig
from stacks.knowledge_base_stack import KnowledgeBaseStack
from stacks.lambda_stack import (
    DOC_SUFFIXES,
    NOTIFICATION_SUFFIXES,
    LambdaStack,
    generation_model_arns,
)
from stacks.s3_vectors_stack import S3VectorsStack


def _build(context: dict, *, with_kb: bool = False) -> dict:
    app = cdk.App(context={"aws:cdk:bundling-stacks": [], "prefix": "testrag", **context})
    cfg = PatternConfig.from_context(app)

    vector_store = S3VectorsStack(app, "vs", cfg=cfg)
    compute = LambdaStack(
        app,
        "comp",
        cfg=cfg,
        vector_bucket_name=vector_store.vector_bucket_name,
        vector_index_name=vector_store.vector_index_name,
        vector_index_arn=vector_store.vector_index_arn,
    )
    built = {"vector_store": vector_store, "compute": compute}
    if with_kb:
        # Mirrors infra/app.py: the KB stack exists only when the flag is on.
        assert cfg.enable_knowledge_base is True
        built["knowledge_base"] = KnowledgeBaseStack(
            app,
            "kb",
            cfg=cfg,
            docs_bucket=compute.docs_bucket,
            vector_bucket_name=vector_store.vector_bucket_name,
            vector_bucket_arn=vector_store.vector_bucket_arn,
        )
    else:
        built["api"] = ApiStack(
            app, "api", cfg=cfg, search_fn=compute.search_fn, query_fn=compute.query_fn
        )
    return {name: Template.from_stack(stack) for name, stack in built.items()}


@pytest.fixture(scope="module")
def templates():
    return _build({})


@pytest.fixture(scope="module")
def kb_templates():
    return _build({"enable_knowledge_base": "true"}, with_kb=True)


def _resources(template: Template, resource_type: str) -> dict:
    return {
        lid: r
        for lid, r in template.to_json()["Resources"].items()
        if r["Type"] == resource_type
    }


def _role_policy_statements(template: Template, role_logical_id: str) -> list[dict]:
    statements = []
    for policy in _resources(template, "AWS::IAM::Policy").values():
        if {"Ref": role_logical_id} in policy["Properties"]["Roles"]:
            statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return statements


def _statement_by_sid(template: Template, sid: str) -> dict:
    for policy in _resources(template, "AWS::IAM::Policy").values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            if statement.get("Sid") == sid:
                return statement
    raise AssertionError(f"no policy statement with Sid {sid}")


def test_vector_bucket_and_index(templates):
    t = templates["vector_store"]
    t.resource_count_is("AWS::S3Vectors::VectorBucket", 1)
    t.has_resource_properties(
        "AWS::S3Vectors::Index",
        {
            "DataType": "float32",
            "Dimension": 1024,
            "DistanceMetric": "cosine",
            "IndexName": "chunks",
            "MetadataConfiguration": {"NonFilterableMetadataKeys": ["text"]},
        },
    )


def test_docs_bucket_is_locked_down(templates):
    # Docs bucket is co-located with the ingestion Lambda in the compute stack.
    templates["compute"].has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": Match.object_like(
                {"BlockPublicAcls": True, "RestrictPublicBuckets": True}
            )
        },
    )


def test_three_python312_arm_functions(templates):
    # The stack also contains CDK helper functions (bucket notifications,
    # auto-delete), so assert on OUR handlers rather than a total count.
    t = templates["compute"]
    for handler in (
        "ingestion.handler.lambda_handler",
        "search.handler.lambda_handler",
        "query.handler.lambda_handler",
    ):
        functions = t.find_resources(
            "AWS::Lambda::Function",
            {
                "Properties": {
                    "Handler": handler,
                    "Runtime": "python3.12",
                    "Architectures": ["arm64"],
                }
            },
        )
        assert len(functions) == 1, f"expected exactly one function for {handler}"


def test_bm25_table_on_demand_with_doc_gsi(templates):
    templates["compute"].has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "GlobalSecondaryIndexes": Match.array_with(
                [Match.object_like({"IndexName": "DocIndex"})]
            ),
        },
    )


def test_ingestion_role_can_write_vectors_but_not_query(templates):
    t = templates["compute"]
    t.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "VectorIndexWrite",
                                    "Action": ["s3vectors:PutVectors", "s3vectors:DeleteVectors", "s3vectors:GetIndex"],
                                }
                            )
                        ]
                    )
                }
            )
        },
    )
    # Query-path roles additionally get GetVectors (required for metadata filters).
    t.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Sid": "VectorIndexQuery",
                                    "Action": ["s3vectors:QueryVectors", "s3vectors:GetVectors", "s3vectors:GetIndex"],
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_api_key_required_and_throttled(templates):
    t = templates["api"]
    methods = t.find_resources("AWS::ApiGateway::Method", {"Properties": {"HttpMethod": "POST"}})
    assert len(methods) == 2
    assert all(m["Properties"]["ApiKeyRequired"] is True for m in methods.values())

    t.has_resource_properties(
        "AWS::ApiGateway::UsagePlan",
        {
            "Throttle": {"BurstLimit": 20, "RateLimit": 10},
            "Quota": {"Limit": 50000, "Period": "MONTH"},
        },
    )
    t.resource_count_is("AWS::ApiGateway::ApiKey", 1)


# --- tenant_id request validation --------------------------------------------


def test_tenant_id_schema_pattern_on_search_and_query(templates):
    models = _resources(templates["api"], "AWS::ApiGateway::Model")
    by_title = {m["Properties"]["Schema"]["title"]: m["Properties"]["Schema"] for m in models.values()}
    for title in ("SearchRequest", "QueryRequest"):
        tenant_schema = by_title[title]["properties"]["tenant_id"]
        assert tenant_schema == {"type": "string", "pattern": TENANT_ID_SCHEMA_PATTERN}


@pytest.mark.parametrize(
    "value",
    [
        "anyexamplecompany", "Anyexamplecompany", "  anyexamplecompany  ", "", "   ", "a" * 64, "a" * 65, "-anyexamplecompany", "_x",
        "a b", "anyexamplecompany!", "ANYEXAMPLECOMPANY_corp-1", "tenant/x", "anyexamplecompany\n",
    ],
)
def test_tenant_id_schema_matches_handler_rule(value):
    """The edge accepts exactly what the handlers normalize to a valid tenant."""
    try:
        resolve_request_tenant(value)
        handler_accepts = True
    except ValueError:
        handler_accepts = False
    assert bool(re.match(TENANT_ID_SCHEMA_PATTERN, value)) is handler_accepts


# --- S3 notifications and the ingestion role ---------------------------------


def test_notifications_cover_created_and_removed_in_both_cases(templates):
    (notifications,) = _resources(templates["compute"], "Custom::S3BucketNotifications").values()
    configs = notifications["Properties"]["NotificationConfiguration"]["LambdaFunctionConfigurations"]
    pairs = {
        (c["Events"][0], c["Filter"]["Key"]["FilterRules"][0]["Value"]) for c in configs
    }
    expected = {
        (event, suffix)
        for event in ("s3:ObjectCreated:*", "s3:ObjectRemoved:*")
        for suffix in NOTIFICATION_SUFFIXES
    }
    assert len(configs) == 24
    assert pairs == expected
    assert {".pdf", ".PDF", ".htm", ".HTM"} <= set(NOTIFICATION_SUFFIXES)


def test_doc_suffixes_match_parser_formats():
    assert {s.lstrip(".") for s in DOC_SUFFIXES} == SUPPORTED_FORMATS


def test_ingestion_role_can_delete_but_not_query(templates):
    t = templates["compute"]
    (ingestion_fn,) = t.find_resources(
        "AWS::Lambda::Function", {"Properties": {"Handler": "ingestion.handler.lambda_handler"}}
    ).values()
    role_id = ingestion_fn["Properties"]["Role"]["Fn::GetAtt"][0]

    actions = set()
    for statement in _role_policy_statements(t, role_id):
        listed = statement["Action"]
        actions.update([listed] if isinstance(listed, str) else listed)

    assert {
        "s3vectors:DeleteVectors",
        "dynamodb:DeleteItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
    } <= actions
    assert "s3vectors:QueryVectors" not in actions


# --- log group outputs ---------------------------------------------------------


@pytest.mark.parametrize(
    "output, construct_id",
    [
        ("IngestionLogGroupName", "IngestionFnLogs"),
        ("SearchLogGroupName", "SearchFnLogs"),
        ("QueryLogGroupName", "QueryFnLogs"),
    ],
)
def test_log_group_names_are_stack_outputs(templates, output, construct_id):
    t = templates["compute"]
    t.has_output(output, {"Value": {"Ref": Match.string_like_regexp(f"^{construct_id}")}})

    logical_id = t.to_json()["Outputs"][output]["Value"]["Ref"]
    log_group = t.to_json()["Resources"][logical_id]
    assert log_group["Type"] == "AWS::Logs::LogGroup"
    assert log_group["Properties"]["RetentionInDays"] == 14
    assert "LogGroupName" not in log_group["Properties"]  # CloudFormation-generated


# --- generation model IAM resources ------------------------------------------

_ARN_PARTS = {"partition": "aws", "region": "us-east-1", "account": "111122223333"}


@pytest.mark.parametrize(
    "model_id, expected",
    [
        (
            "us.anthropic.claude-sonnet-4-6",
            [
                "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-4-6",
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6*",
            ],
        ),
        (
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            [
                "arn:aws:bedrock:us-east-1:111122223333:inference-profile/"
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0*",
            ],
        ),
        (
            "amazon.nova-pro-v1:0",
            ["arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"],
        ),
    ],
)
def test_generation_model_arns(model_id, expected):
    assert generation_model_arns(model_id, **_ARN_PARTS) == expected


def test_default_generation_policy_uses_profile_and_wildcard_foundation_model(templates):
    statement = json.dumps(_statement_by_sid(templates["compute"], "InvokeGenerationModel"))
    assert "inference-profile/us.anthropic.claude-sonnet-4-6" in statement
    assert "foundation-model/anthropic.claude-sonnet-4-6*" in statement


def test_base_model_generation_policy_is_in_region_foundation_model_only():
    compute = _build({"generation_model_id": "amazon.nova-pro-v1:0"})["compute"]
    statement = json.dumps(_statement_by_sid(compute, "InvokeGenerationModel"))
    assert "foundation-model/amazon.nova-pro-v1:0" in statement
    assert "inference-profile" not in statement
    assert ":bedrock:*::" not in statement


# --- optional knowledge base stack (-c enable_knowledge_base=true) -----------


def _kb_role_id(template: Template) -> str:
    (role_id,) = [
        lid
        for lid, role in _resources(template, "AWS::IAM::Role").items()
        if any(
            s.get("Principal", {}).get("Service") == "bedrock.amazonaws.com"
            for s in role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
        )
    ]
    return role_id


def test_kb_data_source_retains_data_on_delete(kb_templates):
    kb_templates["knowledge_base"].has_resource_properties(
        "AWS::Bedrock::DataSource", {"DataDeletionPolicy": "RETAIN"}
    )


def test_kb_depends_on_role_policy(kb_templates):
    t = kb_templates["knowledge_base"]
    role_id = _kb_role_id(t)
    (policy_id,) = [
        lid
        for lid, policy in _resources(t, "AWS::IAM::Policy").items()
        if {"Ref": role_id} in policy["Properties"]["Roles"]
    ]
    (kb,) = _resources(t, "AWS::Bedrock::KnowledgeBase").values()
    assert {policy_id, role_id} <= set(kb["DependsOn"])


def test_kb_index_declares_both_bedrock_non_filterable_keys(kb_templates):
    kb_templates["knowledge_base"].has_resource_properties(
        "AWS::S3Vectors::Index",
        {
            "IndexName": "kb-chunks",
            "MetadataConfiguration": {
                "NonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
            },
        },
    )


def test_kb_role_has_confused_deputy_conditions(kb_templates):
    t = kb_templates["knowledge_base"]
    role = t.to_json()["Resources"][_kb_role_id(t)]
    (statement,) = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    condition = statement["Condition"]
    assert condition["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}
    assert "aws:SourceArn" in condition["ArnLike"]
    assert ":knowledge-base/*" in json.dumps(condition["ArnLike"]["aws:SourceArn"])


# --- ingestion on-failure destination (SQS) ---------------------------------


def _ingestion_fn(template: Template) -> tuple[str, dict]:
    ((fn_id, fn),) = template.find_resources(
        "AWS::Lambda::Function", {"Properties": {"Handler": "ingestion.handler.lambda_handler"}}
    ).items()
    return fn_id, fn


def _failure_queue_id(template: Template) -> str:
    (queue_id,) = _resources(template, "AWS::SQS::Queue")
    assert queue_id.startswith("IngestionFailures")
    return queue_id


def _actions(statement: dict) -> list[str]:
    listed = statement["Action"]
    return [listed] if isinstance(listed, str) else list(listed)


def test_ingestion_failure_queue_is_encrypted_and_retains_14_days(templates):
    t = templates["compute"]
    queue = t.to_json()["Resources"][_failure_queue_id(t)]
    assert queue["Properties"]["SqsManagedSseEnabled"] is True
    assert queue["Properties"]["MessageRetentionPeriod"] == 1209600
    assert queue["DeletionPolicy"] == "Delete"


def test_ingestion_failure_queue_denies_insecure_transport(templates):
    t = templates["compute"]
    queue_id = _failure_queue_id(t)
    (policy,) = _resources(t, "AWS::SQS::QueuePolicy").values()
    assert {"Ref": queue_id} in policy["Properties"]["Queues"]
    deny = [
        s
        for s in policy["Properties"]["PolicyDocument"]["Statement"]
        if s["Effect"] == "Deny"
    ]
    assert any(
        s.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport") == "false"
        for s in deny
    )


def test_ingestion_fn_sends_exhausted_events_to_failure_queue(templates):
    t = templates["compute"]
    fn_id, _ = _ingestion_fn(t)
    queue_id = _failure_queue_id(t)
    (config,) = _resources(t, "AWS::Lambda::EventInvokeConfig").values()
    assert config["Properties"]["FunctionName"] == {"Ref": fn_id}
    assert config["Properties"]["DestinationConfig"]["OnFailure"]["Destination"] == {
        "Fn::GetAtt": [queue_id, "Arn"]
    }
    # Lambda's default retry behavior (two retries) is kept.
    assert "MaximumRetryAttempts" not in config["Properties"]


def test_only_ingestion_role_can_send_to_failure_queue(templates):
    t = templates["compute"]
    _, ingestion_fn = _ingestion_fn(t)
    ingestion_role_id = ingestion_fn["Properties"]["Role"]["Fn::GetAtt"][0]
    queue_arn = {"Fn::GetAtt": [_failure_queue_id(t), "Arn"]}

    send = [
        s
        for s in _role_policy_statements(t, ingestion_role_id)
        if "sqs:SendMessage" in _actions(s)
    ]
    assert send and all(s["Resource"] == queue_arn for s in send)

    for policy in _resources(t, "AWS::IAM::Policy").values():
        if {"Ref": ingestion_role_id} in policy["Properties"]["Roles"]:
            continue
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            assert not any(a.startswith("sqs:") for a in _actions(statement))


def test_ingestion_failure_queue_url_is_stack_output(templates):
    t = templates["compute"]
    t.has_output("IngestionFailureQueueUrl", {"Value": {"Ref": _failure_queue_id(t)}})


# --- API Gateway access logging ------------------------------------------------


def test_api_stage_writes_access_logs_with_api_key_id(templates):
    t = templates["api"]
    (stage,) = _resources(t, "AWS::ApiGateway::Stage").values()
    setting = stage["Properties"]["AccessLogSetting"]
    (log_group_id,) = _resources(t, "AWS::Logs::LogGroup")
    assert log_group_id.startswith("ApiAccessLogs")
    assert setting["DestinationArn"] == {"Fn::GetAtt": [log_group_id, "Arn"]}
    assert "$context.identity.apiKeyId" in setting["Format"]
    assert "$context.requestId" in setting["Format"]
    # Existing stage settings are unchanged.
    assert stage["Properties"]["StageName"] == "prod"
    (method_settings,) = stage["Properties"]["MethodSettings"]
    assert method_settings["MetricsEnabled"] is True


def test_api_access_log_group_retention(templates):
    t = templates["api"]
    (log_group,) = _resources(t, "AWS::Logs::LogGroup").values()
    assert log_group["Properties"]["RetentionInDays"] == 14
    assert log_group["DeletionPolicy"] == "Delete"


def test_api_sets_account_cloudwatch_role_and_retains_it(templates):
    t = templates["api"]
    t.resource_count_is("AWS::ApiGateway::Account", 1)
    (role_id,) = [
        lid
        for lid, role in _resources(t, "AWS::IAM::Role").items()
        if any(
            s.get("Principal", {}).get("Service") == "apigateway.amazonaws.com"
            for s in role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
        )
    ]
    role = t.to_json()["Resources"][role_id]
    assert role["DeletionPolicy"] == "Retain"
    assert "AmazonAPIGatewayPushToCloudWatchLogs" in json.dumps(
        role["Properties"]["ManagedPolicyArns"]
    )
    (account,) = _resources(t, "AWS::ApiGateway::Account").values()
    assert account["Properties"]["CloudWatchRoleArn"] == {"Fn::GetAtt": [role_id, "Arn"]}
