"""Optional managed RAG path: Amazon Bedrock Knowledge Base on S3 Vectors.

Disabled by default. Enable with:
    cdk deploy -c enable_knowledge_base=true --all

Why optional (documented trade-offs):
  * The Knowledge Base manages its OWN vector index and ingestion, so the
    corpus is embedded and stored twice when both paths are active.
  * Verified limitation: Bedrock KB *hybrid* search is NOT supported on
    S3 Vectors - the managed path is semantic-only. The custom BM25+RRF
    pipeline in this pattern exists precisely to close that gap.
  * Uses Titan Text Embeddings V2, available by default in supported
    commercial Regions; no Anthropic model is needed to create the
    knowledge base.

The KB gets a dedicated vector index declaring BOTH Bedrock-managed
non-filterable metadata keys, ``AMAZON_BEDROCK_TEXT`` (chunk text) and
``AMAZON_BEDROCK_METADATA`` (source/chunk bookkeeping) - both required per
the KB prerequisites documentation. Non-filterable keys are immutable after
index creation; changing them means replacing the index (destroy + redeploy
this stack).
"""
from aws_cdk import (
    Aws,
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_bedrock as bedrock,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3vectors as s3vectors,
)
from constructs import Construct

from .config import PatternConfig


class KnowledgeBaseStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: PatternConfig,
        docs_bucket: s3.IBucket,
        vector_bucket_name: str,
        vector_bucket_arn: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        embed_model_arn = (
            f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/{cfg.embedding_model_id}"
        )

        # Dedicated KB-managed index in the shared vector bucket.
        kb_index = s3vectors.CfnIndex(
            self,
            "KbIndex",
            vector_bucket_name=vector_bucket_name,
            index_name=cfg.kb_vector_index_name,
            data_type="float32",
            dimension=cfg.embedding_dimensions,
            distance_metric="cosine",
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                # BOTH keys are required (verified: KB prerequisites doc AND the
                # live failure "Filterable metadata must have at most 2048 bytes").
                # AMAZON_BEDROCK_TEXT holds the chunk text; AMAZON_BEDROCK_METADATA
                # holds Bedrock's managed source/chunk bookkeeping. Keys are
                # immutable after index creation.
                non_filterable_metadata_keys=[
                    "AMAZON_BEDROCK_TEXT",
                    "AMAZON_BEDROCK_METADATA",
                ],
            ),
        )
        kb_index.apply_removal_policy(RemovalPolicy.DESTROY)
        kb_index_arn = f"{vector_bucket_arn}/index/{cfg.kb_vector_index_name}"

        # Service role for the Knowledge Base with confused-deputy protection.
        kb_role = iam.Role(
            self,
            "KbRole",
            assumed_by=iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": Aws.ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}:{Aws.ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            ),
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="EmbeddingModel",
                actions=["bedrock:InvokeModel"],
                resources=[embed_model_arn],
            )
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                sid="VectorIndexAccess",
                actions=[
                    "s3vectors:GetIndex",
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:QueryVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:ListVectors",
                ],
                resources=[kb_index_arn],
            )
        )
        docs_bucket.grant_read(kb_role)

        kb = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name=f"{cfg.prefix}-kb",
            description="Managed RAG path over the same document bucket (semantic-only on S3 Vectors)",
            role_arn=kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embed_model_arn,
                    embedding_model_configuration=bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                        bedrock_embedding_model_configuration=bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                            dimensions=cfg.embedding_dimensions,
                            embedding_data_type="FLOAT32",
                        )
                    ),
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="S3_VECTORS",
                s3_vectors_configuration=bedrock.CfnKnowledgeBase.S3VectorsConfigurationProperty(
                    index_arn=kb_index_arn,
                ),
            ),
        )
        kb.node.add_dependency(kb_index)
        # Bedrock validates index access (s3vectors:QueryVectors) at creation
        # time. Depending on the whole role CONSTRUCT (not just the role ARN
        # reference) also orders the KB after the role's DefaultPolicy - the
        # separate AWS::IAM::Policy resource that actually carries the
        # permissions. Without this, KB creation races the policy and fails
        # with a 403.
        kb.node.add_dependency(kb_role)

        data_source = bedrock.CfnDataSource(
            self,
            "KbDataSource",
            name=f"{cfg.prefix}-kb-docs",
            # RETAIN: on deletion, do NOT try to purge vectors from the index.
            # The kb-chunks index is stack-managed and dies with the stack, so
            # a purge is redundant - and it hard-fails the whole stack delete
            # if the index was ever removed out-of-band (verified live).
            data_deletion_policy="RETAIN",
            knowledge_base_id=kb.attr_knowledge_base_id,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=docs_bucket.bucket_arn,
                ),
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="FIXED_SIZE",
                    fixed_size_chunking_configuration=bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=512,
                        overlap_percentage=15,
                    ),
                ),
            ),
        )

        CfnOutput(self, "KnowledgeBaseId", value=kb.attr_knowledge_base_id)
        CfnOutput(self, "KbDataSourceId", value=data_source.attr_data_source_id)
        CfnOutput(
            self,
            "KbSyncCommand",
            value=(
                "aws bedrock-agent start-ingestion-job "
                f"--knowledge-base-id {kb.attr_knowledge_base_id} "
                f"--data-source-id {data_source.attr_data_source_id}"
            ),
        )
