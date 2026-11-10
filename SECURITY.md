# Security Policy

## Disclaimer

This project is sample code accompanying the AWS Prescriptive Guidance pattern
*Build cost-effective RAG architectures using Amazon S3 Vectors with hybrid search*.
It is provided for educational and demonstration purposes and is
**not intended for production use without additional security hardening**. In particular it uses an
API-key-only, trusted-caller authentication model (see "Known Security
Considerations"). Review and apply the "Production Hardening Recommendations" below
before deploying with untrusted clients or sensitive documents.

## Reporting Vulnerabilities

If you discover a potential security issue in this project, please notify AWS/Amazon
Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/)
or email aws-security@amazon.com. Please do **not** create a public GitHub issue.

## AWS Services Used

- Amazon S3 — source document bucket (`DocsBucket`); S3 event notifications trigger ingestion
- Amazon S3 Vectors — vector bucket and `chunks` index storing Titan embeddings and chunk text
- Amazon DynamoDB — BM25 inverted index, per-tenant statistics, and document manifests
- AWS Lambda — ingestion, search, and query functions (Python 3.12, arm64)
- Amazon API Gateway — REST API (`POST /search`, `POST /query`) with API key, usage plan, throttling, and request validation
- Amazon Bedrock — Amazon Titan Text Embeddings V2 (embeddings) and Anthropic Claude Sonnet 4.6 via the Converse API (generation); optional Amazon Bedrock Knowledge Base on S3 Vectors
- Amazon CloudWatch Logs — function logs (14-day retention) and API access logs
- Amazon SQS — on-failure destination for asynchronous ingestion events
- AWS CloudFormation / AWS CDK — deployment

The API stack sets the account-level Amazon API Gateway CloudWatch Logs role (`cloudWatchRoleArn`), which is one role per account and Region shared by all REST APIs; the role uses the AWS managed policy AmazonAPIGatewayPushToCloudWatchLogs and is retained when you destroy the stack so other APIs keep logging.

## Prerequisites and Permissions

To deploy this solution you need:

- An AWS account with permissions to create S3 buckets, S3 Vectors buckets/indexes,
  DynamoDB tables, Lambda functions and IAM roles, API Gateway REST APIs, SQS queues,
  CloudWatch log groups, and (optionally) Bedrock Knowledge Bases — in practice the
  CDK bootstrap deployment role
- Amazon Bedrock model access in the deployment Region for `amazon.titan-embed-text-v2:0`
  and the Claude Sonnet 4.6 cross-Region inference profile (one-time Anthropic use-case
  form and a first test call from an identity with `aws-marketplace:Subscribe`)
- AWS CLI v2, Python 3.12, Node.js 22+ (CDK CLI), and a container runtime (Docker,
  Colima, or Finch) for Lambda asset bundling

## Known Security Considerations

The following trade-offs were accepted during the AWS Public Content Security Review
for this sample and should be revisited before any production use:

| Item | Category | Rationale |
|------|----------|-----------|
| API key is the only authentication; `tenant_id` is supplied by the caller (trusted-caller model) | Security Debt | Keeps the sample focused on retrieval. Any holder of the shared key can query any tenant. See `docs/architecture.md` → "Multi-tenant isolation" for the authorizer swap. |
| CORS allows all origins; no AWS WAF | Security Debt | Simplifies testing from any client. A browser client would have to embed the API key. |
| No Amazon Bedrock Guardrails; retrieved document text is placed in the prompt | Security Debt | Documents are assumed trusted. A crafted document can influence the answer for its own tenant (indirect prompt injection); the model has no tools, so there are no side effects. |
| AWS-owned / SSE-S3 encryption keys (no customer-managed KMS key) | Security Debt | All data is encrypted at rest; logs contain identifiers and counts only. CMK adds key-policy, rotation, and CloudTrail auditability. |
| No S3 server access logging on the document bucket | Security Debt | Uploads/deletes are visible in the ingestion function log; principal attribution is not. |
| `RemovalPolicy.DESTROY`, `auto_delete_objects`, no versioning, no PITR | Security Debt | Lets `cdk destroy --all` clean up completely. Indexes are rebuildable from the source bucket. |
| No resource tags; no reserved Lambda concurrency | Security Debt | Usage-plan throttling bounds the API path; ingestion scales with uploads. |

## Production Hardening Recommendations

- **Authentication/authorization**: add a Cognito user pool or IAM authorizer to both
  API methods; derive `tenant_id` from the authenticated identity
  (`resolve_request_tenant()` in `src/common/tenant.py` is the single swap point) and
  reject body-supplied tenants; restrict CORS `allow_origins` to your application
  domain; attach an AWS WAF web ACL (Common Rule Set + rate-based rule) to the stage.
- **Upload authorization**: scope uploader IAM policies with an `s3:prefix` /
  object-key condition so each principal can write only under its own
  `tenants/<tenant_id>/` prefix.
- **GenAI safety**: create an Amazon Bedrock Guardrail (content filters incl.
  PROMPT_ATTACK, sensitive-information policy, contextual grounding) and pass
  `guardrailConfig` to `converse()`; wrap context passages in a delimiter the system
  prompt declares as untrusted data.
- **Encryption**: create a customer-managed KMS key with rotation and use it for the
  DynamoDB table, all log groups, Lambda environment variables, the document bucket,
  and the S3 Vectors bucket; grant decrypt to the function and Knowledge Base roles.
- **Logging and audit**: enable S3 server access logging (or CloudTrail S3 data
  events) on the document bucket; keep API Gateway access logging enabled and include
  `$context.identity.apiKeyId`; raise log retention to your compliance period.
- **Durability**: `RemovalPolicy.RETAIN` and `versioned=True` on the document bucket,
  `point_in_time_recovery=True` and `deletion_protection=True` on the DynamoDB table;
  remove `auto_delete_objects`.
- **Operations**: add `Tags.of(app)` for ownership/cost allocation; set
  `reserved_concurrent_executions` on the ingestion function sized to your Bedrock
  embeddings quota; alarm on the ingestion failure queue depth.
- **Dependencies**: keep `pypdf` current (≥ 6.21.4 at review time); pin with hashes
  (`pip-compile --generate-hashes`) and run `pip-audit` in CI.

## Resource Cleanup

1. `cd infra && cdk destroy --all` (add `-c enable_knowledge_base=true` if the
   Knowledge Base stack was deployed). The document bucket is emptied automatically;
   the DynamoDB table, vector indexes, vector bucket, log groups, and failure queue are
   deleted with their stacks.
2. If the vector bucket refuses to delete because an index was created outside these
   stacks: `aws s3vectors delete-index --vector-bucket-name <name> --index-name <name>`,
   then run destroy again.
3. Optionally delete the CDK bootstrap stack (`CDKToolkit`) if nothing else uses it.

## Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| boto3 | 1.40.0 | AWS SDK; pinned for the `s3vectors` client and Converse features. No known advisories. |
| pypdf | 6.21.4 | PDF text extraction for uploaded documents. Parses untrusted input — keep current; 5.x had multiple DoS CVEs. |
| aws-cdk-lib | 2.268.0 | CDK v2 (needs ≥ 2.230.0 for S3 Vectors KB storage). |
| constructs | 10.8.1 | CDK constructs library. |
| pytest | 9.2.0 | Test runner (dev only, not bundled). |
