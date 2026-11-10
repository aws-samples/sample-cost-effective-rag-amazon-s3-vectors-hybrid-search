"""Vector storage stack.

Creates:
  * An S3 *vector bucket* (AWS::S3Vectors::VectorBucket) - purpose-built,
    low-cost vector storage.
  * A *vector index* (AWS::S3Vectors::Index) - cosine distance, float32,
    dimensions matching Titan Text Embeddings V2 (1024 by default).
    The chunk text is stored as NON-FILTERABLE metadata under the "text" key,
    so similarity queries can return the passage without a second lookup.
The source-document S3 bucket lives in the compute stack alongside the
ingestion Lambda: S3 event notifications are synthesized into the bucket's
stack and reference the target function, so co-locating them avoids a
cross-stack dependency cycle.

Notes verified against the S3 Vectors documentation:
  * Filterable metadata (tenant_id, doc_id, source, ...) is limited to 2 KB
    per vector; non-filterable keys are configured at index creation and
    cannot be changed later (max 10 per index).
  * A vector bucket can be deleted only when it has no indexes. Both indexes
    (``chunks`` here, ``kb-chunks`` in the knowledge base stack) are
    stack-managed with RemovalPolicy.DESTROY, so `cdk destroy --all` removes
    them first; don't delete them manually.
"""
from aws_cdk import (
    Aws,
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_s3vectors as s3vectors,
)
from constructs import Construct

from .config import PatternConfig


class S3VectorsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, cfg: PatternConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Vector bucket (explicit name so Lambdas can address it via env vars
        # and IAM ARNs can be constructed deterministically).
        # Naming rules: 3-63 chars, lowercase letters, numbers, hyphens.
        # ------------------------------------------------------------------
        self.vector_bucket_name = f"{cfg.prefix}-vectors-{Aws.ACCOUNT_ID}"
        self.vector_index_name = cfg.vector_index_name

        vector_bucket = s3vectors.CfnVectorBucket(
            self,
            "VectorBucket",
            vector_bucket_name=self.vector_bucket_name,
            # Default SSE-S3 (AES256) encryption applies when not specified.
        )
        vector_bucket.apply_removal_policy(RemovalPolicy.DESTROY)
        self.vector_bucket_arn = (
            f"arn:{Aws.PARTITION}:s3vectors:{Aws.REGION}:{Aws.ACCOUNT_ID}"
            f":bucket/{self.vector_bucket_name}"
        )

        # ------------------------------------------------------------------
        # Vector index for the custom hybrid-search pipeline.
        # ------------------------------------------------------------------
        index = s3vectors.CfnIndex(
            self,
            "ChunkIndex",
            vector_bucket_name=self.vector_bucket_name,
            index_name=self.vector_index_name,
            data_type="float32",
            dimension=cfg.embedding_dimensions,
            distance_metric="cosine",  # recommended metric for Titan V2 embeddings
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                # "text" holds the chunk passage (up to ~2 KB chunks). It is
                # returned with query results but cannot be used in filters.
                non_filterable_metadata_keys=["text"],
            ),
        )
        index.node.add_dependency(vector_bucket)
        index.apply_removal_policy(RemovalPolicy.DESTROY)

        # ARN format (verified): arn:aws:s3vectors:REGION:ACCOUNT:bucket/NAME/index/INDEX
        self.vector_index_arn = f"{self.vector_bucket_arn}/index/{self.vector_index_name}"

        CfnOutput(self, "VectorBucketName", value=self.vector_bucket_name)
        CfnOutput(self, "VectorIndexName", value=self.vector_index_name)
        CfnOutput(self, "VectorIndexArn", value=self.vector_index_arn)
