#!/usr/bin/env python3
"""CDK application entry point for the S3 Vectors hybrid RAG pattern.

Stacks:
    <prefix>-vector-store   S3 vector bucket + vector index
    <prefix>-compute        Document bucket, Lambda functions, DynamoDB BM25
                            table, IAM
    <prefix>-api            REST API with API key, usage plan, and throttling
    <prefix>-knowledge-base (optional) Bedrock Knowledge Base on S3 Vectors,
                            enabled with: cdk deploy -c enable_knowledge_base=true
"""
import os

import aws_cdk as cdk

from stacks.config import PatternConfig
from stacks.s3_vectors_stack import S3VectorsStack
from stacks.lambda_stack import LambdaStack
from stacks.api_stack import ApiStack
from stacks.knowledge_base_stack import KnowledgeBaseStack

app = cdk.App()
cfg = PatternConfig.from_context(app)

# S3 Vectors is available in most commercial Regions (see the S3 Vectors
# Regions page). The app uses CDK_DEFAULT_REGION (set by the CDK CLI from your
# profile) and falls back to us-east-1.
env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION") or "us-east-1",
)

vector_store = S3VectorsStack(
    app,
    f"{cfg.prefix}-vector-store",
    cfg=cfg,
    env=env,
    description="S3 Vectors hybrid RAG: vector bucket and vector index",
)

compute = LambdaStack(
    app,
    f"{cfg.prefix}-compute",
    cfg=cfg,
    vector_bucket_name=vector_store.vector_bucket_name,
    vector_index_name=vector_store.vector_index_name,
    vector_index_arn=vector_store.vector_index_arn,
    env=env,
    description="S3 Vectors hybrid RAG: docs bucket, ingestion/search/query Lambdas, BM25 table",
)
# Names/ARNs above resolve via pseudo parameters (no CFN reference), so make
# the deploy ordering explicit: the vector index must exist before Lambdas run.
compute.add_stack_dependency(vector_store)

api = ApiStack(
    app,
    f"{cfg.prefix}-api",
    cfg=cfg,
    search_fn=compute.search_fn,
    query_fn=compute.query_fn,
    env=env,
    description="S3 Vectors hybrid RAG: REST API with usage plan and throttling",
)

if cfg.enable_knowledge_base:
    kb = KnowledgeBaseStack(
        app,
        f"{cfg.prefix}-knowledge-base",
        cfg=cfg,
        docs_bucket=compute.docs_bucket,
        vector_bucket_name=vector_store.vector_bucket_name,
        vector_bucket_arn=vector_store.vector_bucket_arn,
        env=env,
        description="S3 Vectors hybrid RAG (optional managed path): Bedrock Knowledge Base on S3 Vectors",
    )
    kb.add_stack_dependency(vector_store)

app.synth()
