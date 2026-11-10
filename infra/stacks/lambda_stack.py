"""Compute stack: document bucket, Lambda functions, DynamoDB BM25 table, IAM.

Design decisions:
  * The source-document bucket lives HERE (not in the vector-store stack):
    S3 event notifications are synthesized into the bucket's stack and
    reference the target Lambda, so bucket + functions must be co-located to
    avoid a cross-stack dependency cycle.
  * One deployment asset built from ``src/`` is shared by all three functions
    (handlers: ingestion.handler / search.handler / query.handler). The
    ``common`` package and the search modules are therefore importable by the
    query function in-process - no Lambda-to-Lambda hop on the query path.
  * Asset bundling is Docker-only: ``pip install`` runs inside the
    Lambda-compatible Python 3.12 build image, pinned to the functions'
    architecture (``linux/arm64``), so dependencies always match the runtime
    regardless of the build host. ``cdk synth``/``cdk deploy`` therefore need
    a running container runtime (Docker, Colima, or Finch via
    ``CDK_DOCKER=finch``).
  * The ingestion function handles both ObjectCreated (index the document)
    and ObjectRemoved (remove its vectors and BM25 postings) notifications.
  * Log groups keep CloudFormation-generated names; the stack exports them
    as outputs (IngestionLogGroupName, SearchLogGroupName, QueryLogGroupName).
  * IAM follows least privilege, including the S3 Vectors subtlety that
    metadata filtering / returnMetadata additionally requires
    ``s3vectors:GetVectors``, and the Bedrock rule that a cross-Region
    inference profile needs BOTH the inference-profile ARN and the underlying
    foundation-model ARN with a wildcard Region (a base model ID needs only
    its in-Region foundation-model ARN; see ``generation_model_arns``).
"""
from pathlib import Path

from aws_cdk import (
    BundlingOptions,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Aws,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_lambda_destinations as destinations,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_sqs as sqs,
)
from constructs import Construct

from .config import PatternConfig

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
# Keep in sync with src/ingestion/parsers.py SUPPORTED_FORMATS - objects with
# other suffixes never trigger the ingestion function.
DOC_SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".html", ".htm")
# S3 suffix filters are case-sensitive, so each suffix is registered in
# lowercase and uppercase (.pdf and .PDF). Mixed case (for example .Pdf)
# isn't matched, and such objects never trigger the ingestion function.
NOTIFICATION_SUFFIXES = DOC_SUFFIXES + tuple(s.upper() for s in DOC_SUFFIXES)

# Model ID prefixes of geographic and global cross-Region inference profiles.
# us., eu., apac., in. (geographic cross-Region inference page) and au., jp.,
# global. (Claude Haiku 4.5 model card) were confirmed on the Amazon Bedrock
# docs on 2026-10-12. us-gov. and ca. weren't found on the pages checked and
# are kept defensively: no model provider ID starts with them.
INFERENCE_PROFILE_PREFIXES = ("global.", "us.", "us-gov.", "eu.", "apac.", "au.", "jp.", "in.", "ca.")


def generation_model_arns(model_id: str, *, partition: str, region: str, account: str) -> list[str]:
    """IAM resources that allow invoking ``model_id`` with Converse/InvokeModel.

    * Cross-Region inference profile ID (``us.anthropic.claude-sonnet-4-6``):
      the account-local inference-profile ARN, plus the foundation-model ARN
      of the underlying model (profile prefix stripped) with a wildcard
      Region, because the request may be routed to any Region in the profile.
    * Base model ID (``amazon.nova-pro-v1:0``): only the foundation-model ARN
      in this Region.

    Sources: Amazon Bedrock User Guide, "Geographic cross-Region inference"
    (IAM policy requirements) and the per-model cards for profile IDs.
    """
    for prefix in INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            base_model = model_id[len(prefix):]
            return [
                f"arn:{partition}:bedrock:{region}:{account}:inference-profile/{model_id}",
                f"arn:{partition}:bedrock:*::foundation-model/{base_model}*",
            ]
    return [f"arn:{partition}:bedrock:{region}::foundation-model/{model_id}"]


class LambdaStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: PatternConfig,
        vector_bucket_name: str,
        vector_index_name: str,
        vector_index_arn: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Source-document bucket (standard S3), co-located with its consumer.
        # ------------------------------------------------------------------
        self.docs_bucket = s3.Bucket(
            self,
            "DocsBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,  # sample-friendly cleanup
        )
        docs_bucket = self.docs_bucket

        # ------------------------------------------------------------------
        # DynamoDB table: BM25 inverted index + per-tenant stats + manifests.
        #   Posting : PK=T#<tenant>#TERM#<term>   SK=CHUNK#<chunk_key>  (tf, dl)
        #   Stats   : PK=T#<tenant>#STATS         SK=GLOBAL             (doc_count, total_len)
        #   Manifest: PK=T#<tenant>#DOC#<doc_id>  SK=MANIFEST           (chunk_count, total_len)
        #   GSI DocIndex (gsi1pk = T#<tenant>#DOC#<doc_id>, KEYS_ONLY) locates a
        #   document's postings for cleanup on re-ingestion.
        # ------------------------------------------------------------------
        table = dynamodb.Table(
            self,
            "Bm25Table",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )
        table.add_global_secondary_index(
            index_name="DocIndex",
            partition_key=dynamodb.Attribute(name="gsi1pk", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.KEYS_ONLY,
        )
        self.table = table

        # ------------------------------------------------------------------
        # Shared Lambda code asset (src/ + pip deps), bundled in a container
        # pinned to the functions' runtime and architecture.
        # ------------------------------------------------------------------
        runtime = _lambda.Runtime.PYTHON_3_12
        architecture = _lambda.Architecture.ARM_64
        code = _lambda.Code.from_asset(
            str(SRC_DIR),
            bundling=BundlingOptions(
                image=runtime.bundling_image,  # public.ecr.aws/sam/build-python3.12
                platform=architecture.docker_platform,  # linux/arm64
                command=[
                    "bash",
                    "-c",
                    "pip install -r requirements.txt -t /asset-output --quiet "
                    "&& cp -au . /asset-output",
                ],
            ),
        )

        common_env = {
            "VECTOR_BUCKET_NAME": vector_bucket_name,
            "VECTOR_INDEX_NAME": vector_index_name,
            "BM25_TABLE_NAME": table.table_name,
            "EMBEDDING_MODEL_ID": cfg.embedding_model_id,
            "EMBEDDING_DIMENSIONS": str(cfg.embedding_dimensions),
            "GENERATION_MODEL_ID": cfg.generation_model_id,
            "DEFAULT_TENANT": "default",
            "MAX_CONTEXT_TOKENS": "6000",
            "CHUNK_CHARS": "2048",
            "CHUNK_OVERLAP_CHARS": "256",
            "LOG_LEVEL": "INFO",
        }

        # On-failure destination for asynchronous S3 events that exhaust
        # Lambda's retries (two by default) on the ingestion function.
        ingestion_failures = sqs.Queue(
            self,
            "IngestionFailures",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
            removal_policy=RemovalPolicy.DESTROY,
        )

        def make_fn(
            name: str,
            handler: str,
            memory: int,
            timeout_s: int,
            *,
            on_failure_queue: sqs.IQueue | None = None,
        ) -> tuple[_lambda.Function, logs.LogGroup]:
            # No log_group_name: CloudFormation generates the name, exported
            # below as a stack output.
            log_group = logs.LogGroup(
                self,
                f"{name}Logs",
                retention=logs.RetentionDays.TWO_WEEKS,
                removal_policy=RemovalPolicy.DESTROY,
            )
            fn = _lambda.Function(
                self,
                name,
                runtime=runtime,
                architecture=architecture,
                code=code,
                handler=handler,
                memory_size=memory,
                timeout=Duration.seconds(timeout_s),
                environment=dict(common_env),
                log_group=log_group,
                on_failure=(
                    destinations.SqsDestination(on_failure_queue) if on_failure_queue else None
                ),
            )
            return fn, log_group

        self.ingestion_fn, ingestion_logs = make_fn(
            "IngestionFn",
            "ingestion.handler.lambda_handler",
            1024,
            300,
            on_failure_queue=ingestion_failures,
        )
        self.search_fn, search_logs = make_fn("SearchFn", "search.handler.lambda_handler", 512, 30)
        self.query_fn, query_logs = make_fn("QueryFn", "query.handler.lambda_handler", 1024, 60)

        # ------------------------------------------------------------------
        # S3 -> ingestion trigger for each supported document suffix, on
        # upload (index the document) and on deletion (remove it from both
        # indexes).
        # ------------------------------------------------------------------
        for event_type in (s3.EventType.OBJECT_CREATED, s3.EventType.OBJECT_REMOVED):
            for suffix in NOTIFICATION_SUFFIXES:
                docs_bucket.add_event_notification(
                    event_type,
                    s3n.LambdaDestination(self.ingestion_fn),
                    s3.NotificationKeyFilter(suffix=suffix),
                )

        # ------------------------------------------------------------------
        # IAM - least privilege per function.
        # ------------------------------------------------------------------
        embed_model_arn = (
            f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/{cfg.embedding_model_id}"
        )

        invoke_embedding = iam.PolicyStatement(
            sid="InvokeEmbeddingModel",
            actions=["bedrock:InvokeModel"],
            resources=[embed_model_arn],
        )
        invoke_generation = iam.PolicyStatement(
            sid="InvokeGenerationModel",
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=generation_model_arns(
                cfg.generation_model_id,
                partition=Aws.PARTITION,
                region=Aws.REGION,
                account=Aws.ACCOUNT_ID,
            ),
        )

        # Ingestion: read docs, write vectors, write BM25 postings. The same
        # grants cover deletions (s3vectors:DeleteVectors below; DynamoDB
        # Query/DeleteItem/UpdateItem from grant_read_write_data) and the
        # HeadObject re-upload check (s3:GetObject from grant_read).
        docs_bucket.grant_read(self.ingestion_fn)
        table.grant_read_write_data(self.ingestion_fn)
        self.ingestion_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="VectorIndexWrite",
                actions=[
                    "s3vectors:PutVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:GetIndex",
                ],
                resources=[vector_index_arn],
            )
        )
        self.ingestion_fn.add_to_role_policy(invoke_embedding)

        # Search: query vectors (+GetVectors for filters/metadata), read BM25,
        # embed the query string.
        table.grant_read_data(self.search_fn)
        self.search_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="VectorIndexQuery",
                actions=[
                    "s3vectors:QueryVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:GetIndex",
                ],
                resources=[vector_index_arn],
            )
        )
        self.search_fn.add_to_role_policy(invoke_embedding)

        # Query: everything search needs (hybrid search runs in-process),
        # plus the generation model.
        table.grant_read_data(self.query_fn)
        self.query_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="VectorIndexQuery",
                actions=[
                    "s3vectors:QueryVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:GetIndex",
                ],
                resources=[vector_index_arn],
            )
        )
        self.query_fn.add_to_role_policy(invoke_embedding)
        self.query_fn.add_to_role_policy(invoke_generation)

        CfnOutput(self, "DocsBucketName", value=self.docs_bucket.bucket_name)
        CfnOutput(self, "Bm25TableName", value=table.table_name)
        CfnOutput(self, "IngestionFunctionName", value=self.ingestion_fn.function_name)
        CfnOutput(self, "SearchFunctionName", value=self.search_fn.function_name)
        CfnOutput(self, "QueryFunctionName", value=self.query_fn.function_name)
        CfnOutput(self, "IngestionLogGroupName", value=ingestion_logs.log_group_name)
        CfnOutput(self, "SearchLogGroupName", value=search_logs.log_group_name)
        CfnOutput(self, "QueryLogGroupName", value=query_logs.log_group_name)
        CfnOutput(
            self,
            "IngestionFailureQueueUrl",
            value=ingestion_failures.queue_url,
            description="SQS queue that receives S3 events the ingestion function "
            "failed to process after all retries",
        )
