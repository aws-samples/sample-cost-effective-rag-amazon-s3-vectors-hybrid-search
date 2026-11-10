# Build cost-effective RAG architectures using Amazon S3 Vectors with hybrid search

Cost-effective retrieval augmented generation (RAG) with **hybrid search** on
**Amazon S3 Vectors**. This repository contains the sample code for the AWS
Prescriptive Guidance (APG) pattern *Build cost-effective RAG architectures
using Amazon S3 Vectors with hybrid search*.

> **Sample code — not for production use as-is.** This project demonstrates a pattern and
> intentionally accepts trade-offs (API-key-only authentication with a trusted-caller
> tenant model, no Bedrock Guardrails, AWS-owned encryption keys). Review
> [SECURITY.md](SECURITY.md) and apply the production hardening recommendations before
> deploying with untrusted clients or sensitive documents.

## What it does

Drop documents into an S3 bucket; ask questions over them through a REST API;
get back grounded answers with citations. Concretely:

- **Ingests documents automatically** - upload a PDF, TXT, MD, or HTML file
  to the document bucket and it is parsed, chunked, embedded, and indexed
  within seconds. Re-uploading the same file re-indexes it cleanly, and
  deleting it removes its vectors and BM25 postings. File suffixes match in
  all-lowercase or all-uppercase form (`.pdf` or `.PDF`, not `.Pdf`).
- **Answers questions** (`POST /query`) - retrieves the most relevant
  passages and has Claude Sonnet 4.6 compose an answer that cites them. If
  the documents don't contain the answer, it says so instead of inventing one.
- **Exposes raw retrieval** (`POST /search`) - returns ranked passages
  without calling the LLM, in three modes: `dense` (meaning-based), `sparse`
  (keyword/BM25), or `hybrid` (both, fused). Useful for debugging relevance
  and building your own downstream logic.
- **Isolates tenants** - documents uploaded under `tenants/<id>/...` are only
  ever retrievable by queries carrying that `tenant_id`, enforced in the
  storage layer on both retrieval paths. `tenant_id` must match
  `^[a-z0-9][a-z0-9_-]{0,63}$` after trimming and lowercasing; blank or
  omitted means the default tenant.
- **Costs almost nothing at rest** - every component is serverless and
  pay-per-request, with no idle compute; the vector store's storage costs
  cents per month at 100k chunks, where an Amazon OpenSearch Serverless
  classic collection starts at about $175/month (dev-test) or about
  $350/month (high availability) (see [Cost estimate](#cost-estimate-us-east-1-list-prices-retrieved-october-12-2026)).

Why this pattern exists: Amazon Bedrock Knowledge Bases hybrid search is
**not supported on S3 Vectors** (semantic-only). This project brings hybrid
retrieval to Amazon S3 Vectors. For the full write-up, see the APG pattern.

## How it works

```
S3 upload / delete ──▶ Ingestion λ ──▶ Titan V2 ──▶ S3 Vectors (dense + chunk text)
                            └────────────────────▶ DynamoDB   (BM25 postings)

client ──▶ API GW ──▶ /search λ ─┬▶ dense top-k (tenant filtered)
                      /query  λ ─┼▶ sparse BM25 top-k
                                 └▶ RRF(k=60) ──▶ Claude Sonnet 4.6 ──▶ answer + citations
```

### Ingestion pipeline (per uploaded document)

1. **Trigger** - an S3 `ObjectCreated` event invokes the ingestion
   AWS Lambda function. Ingestion triggers on `.pdf .txt .md .markdown .html .htm` in
   all-lowercase or all-uppercase form (for example, `.pdf` or `.PDF`); S3
   suffix filters are case-sensitive, so mixed-case suffixes such as `.Pdf`
   don't trigger ingestion. The tenant comes from the key prefix
   (`tenants/anyexamplecompany/...` -> `anyexamplecompany`); the folder name is lowercased and must
   match `^[a-z0-9][a-z0-9_-]{0,63}$`, otherwise the upload isn't ingested
   (`document_rejected` log record). Deleting a document from the bucket
   invokes the ingestion function through an `s3:ObjectRemoved`
   notification, which uses the document's manifest to delete its vectors
   and BM25 postings and decrement the tenant statistics
   (`document_deleted` log record).
2. **Parse** - text is extracted (pypdf for PDF; stdlib parsers for
   HTML/TXT/MD).
3. **Chunk** - recursive character splitting into ~512-token chunks
   (~2,048 chars) with ~12.5% overlap, preferring paragraph, line, then
   sentence boundaries.
4. **Embed** - each chunk goes to Amazon Titan Text Embeddings V2 and comes back as
   a 1,024-dimension vector.
5. **Index (dense)** - vectors are written to the S3 Vectors index in
   batches. Filterable metadata: `tenant_id`, `doc_id`, `source`,
   `chunk_index`, `format`, `ingested_at`. The chunk text itself rides along
   as *non-filterable* metadata, so search results carry their passage with
   no second lookup.
6. **Index (sparse)** - the same chunks are tokenized (lowercase, stopwords
   removed, light stemming) into BM25 postings in Amazon DynamoDB:
   `term -> (chunk, term frequency, chunk length)`, plus per-tenant
   statistics. A manifest item records what was written so a re-upload
   deletes the old vectors and postings first (idempotent re-ingestion).

### Query pipeline (per question)

1. **Embed the question** with the same Titan model.
2. **Dense search** - `QueryVectors` on the S3 Vectors index (cosine
   similarity), always with a `tenant_id` metadata filter and at most 100
   candidates per query (the QueryVectors page size). Filter timing depends
   on the index mode: ENHANCED (default for vector buckets created on or
   after September 30, 2026) pre-filters; CLASSIC filters during the search
   and can return fewer than top-k.
3. **Sparse search** - the question's terms are looked up in the DynamoDB
   inverted index and chunks are scored with Okapi BM25 (k1=1.2, b=0.75).
4. **Fuse** - both ranked lists merge via Reciprocal Rank Fusion:
   `score = Σ 1/(60 + rank)`. Rank-only fusion needs no calibration between
   cosine and BM25 scales. A chunk found by both paths beats a chunk found
   by one.
5. **Build the prompt** - top chunks are packed into a numbered context
   block under a token budget (default 6,000), with a system prompt that
   demands citations and forbids answering beyond the context.
6. **Generate** - Claude Sonnet 4.6 via the Bedrock Converse API (explicit
   `maxTokens`, adaptive retries, cross-Region inference profile). The
   response returns the answer, per-passage citations, and token usage.

Why two retrieval paths? Embeddings blur exact identifiers (error codes,
SKUs, names) that keyword search nails; keyword search returns nothing for
paraphrases ("everything feels sluggish") that embeddings handle easily.
Hybrid covers both failure modes. Try it yourself: run the same `/search`
query in each mode and compare rankings.

## API reference

Both endpoints require the `x-api-key` header.

### POST /search - retrieval only

Request:

```json
{
  "query": "ORN-429 rate limit",     // required
  "tenant_id": "anyexamplecompany",               // optional; trimmed and lowercased; must match ^[a-z0-9][a-z0-9_-]{0,63}$; omitted or blank = "default"
  "top_k": 5,                        // optional, 1-50, default 5
  "mode": "hybrid"                   // optional: hybrid | dense | sparse
}
```

Response (`results` ordered best-first):

```json
{
  "query": "...", "tenant_id": "anyexamplecompany", "mode": "hybrid",
  "results": [{
    "key": "anyexamplecompany#example-corp-platform-runbook-3bbe9802#00001",
    "score": 0.0325,                 // RRF score (hybrid), 1-distance (dense), BM25 (sparse)
    "text": "...the chunk passage...",
    "metadata": {"doc_id": "...", "source": "tenants/anyexamplecompany/...", "chunk_index": 1},
    "dense_rank": 1,                 // null if this path didn't surface the chunk
    "sparse_rank": 2
  }]
}
```

### POST /query - retrieval + grounded generation

Request:

```json
{
  "question": "What does ORN-601 mean?",   // required
  "tenant_id": "anyexamplecompany",                     // optional; trimmed and lowercased; must match ^[a-z0-9][a-z0-9_-]{0,63}$; omitted or blank = "default"
  "top_k": 5,                              // optional, 1-20
  "max_tokens": 1024,                      // optional, cap 4096
  "temperature": 0.2                       // optional, 0-1
}
```

Response:

```json
{
  "answer": "ORN-601 indicates a settlement batch validation failure... [1]",
  "citations": [{"index": 1, "key": "...", "doc_id": "...", "source": "...",
                 "score": 0.0325, "dense_rank": 1, "sparse_rank": 2}],
  "model": "us.anthropic.claude-sonnet-4-6",
  "usage": {"inputTokens": 1813, "outputTokens": 138, "totalTokens": 1951}
}
```

Errors: `400` for invalid input (including a `tenant_id` that doesn't match
the pattern), `403` for a missing/wrong API key, `429` when throttled, `500`
with a generic message on internal failure (details go to Amazon CloudWatch
Logs, never to the caller).

## Project structure

```
infra/                    CDK v2 app (Python)
  app.py                  entry point (stacks: vector-store, compute, api, [kb])
  stacks/                 s3_vectors_stack, lambda_stack, api_stack, knowledge_base_stack
src/                      Lambda code (Python 3.12, one shared bundle)
  common/                 config, cached boto3 clients (adaptive retry), tokenizer, tenant ID rule
  ingestion/              handler, parsers (pdf/txt/md/html), chunker, embedder, bm25_indexer
  search/                 handler, dense_search, sparse_search (BM25), rrf_fusion, hybrid
  query/                  handler (Converse, maxTokens explicit), prompt_builder (token budget)
scripts/                  smoke_test.py (live E2E verification)
samples/                  example-corp-platform-runbook.md - realistic test document
tests/                    pytest: unit tests + CDK assertions
docs/                     architecture (Mermaid + draw.io diagram), cost-comparison, migration-guide
```

## Prerequisites

- AWS account + credentials, AWS CLI v2
- Python 3.12, Node.js 22.x or later (24.x LTS recommended) for the CDK CLI
  (`npm install -g aws-cdk`)
- Amazon Bedrock access in the deployment Region: Titan Text Embeddings V2
  and Claude Sonnet 4.6 are available by default. Anthropic models need the
  one-time use-case form, and the first call needs `aws-marketplace:Subscribe`
  from an identity that has it (the Lambda roles don't), so make one test
  call from the Amazon Bedrock console playground or the AWS CLI before the
  first `/query`
- Docker, or a compatible container runtime such as Colima or Finch (set
  `CDK_DOCKER=finch`), running during `cdk synth` and `cdk deploy` - required.
  CDK installs the Lambda dependencies in a Lambda-compatible Python 3.12
  ARM64 container (`linux/arm64`). `pytest` does not need it.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# run the test suite (no AWS needed)
pytest

# start the container runtime CDK bundles the Lambda asset with
# (Docker Desktop, `colima start`, or `finch vm start && export CDK_DOCKER=finch`)
colima start
docker info            # confirm the daemon is reachable

# deploy (to your AWS CLI profile's Region; us-east-1 if none is set)
cd infra
cdk bootstrap          # first time in the account/region
cdk deploy --all       # add: -c enable_knowledge_base=true for the managed KB path
cd ..

# verify the live deployment end to end (upload -> index -> search -> answer -> isolation)
python scripts/smoke_test.py
```

Then explore with the bundled sample document:

```bash
# outputs from the deploy: ApiEndpoint, ApiKeyId, DocsBucketName, IngestionLogGroupName
API_KEY=$(aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query value --output text)

# 1. ingest the sample runbook as tenant "anyexamplecompany"
aws s3 cp samples/example-corp-platform-runbook.md "s3://<DocsBucketName>/tenants/anyexamplecompany/example-corp-platform-runbook.md"

# 2. raw retrieval - try mode dense vs sparse vs hybrid and compare rankings
curl -s -X POST "<ApiEndpoint>search" -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "ORN-429 rate limit", "tenant_id": "anyexamplecompany", "top_k": 3, "mode": "hybrid"}'

# 3. grounded answer with citations
curl -s -X POST "<ApiEndpoint>query" -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the p99 latency SLO for authorizations?", "tenant_id": "anyexamplecompany"}'

# 4. check the ingestion logs (look for document_ingested, document_deleted, document_rejected)
aws logs tail <IngestionLogGroupName> --since 10m

# 5. delete the document (its vectors and postings are removed too)
aws s3 rm "s3://<DocsBucketName>/tenants/anyexamplecompany/example-corp-platform-runbook.md"
```

## Configuration (CDK context, `infra/cdk.json` or `-c key=value`)

| Key | Default | Notes |
| --- | --- | --- |
| `generation_model_id` | `us.anthropic.claude-sonnet-4-6` | cross-Region profile; cost knob: `us.anthropic.claude-haiku-4-5-20251001-v1:0` (per-token price one-third of Claude Sonnet 4.6; Claude Haiku 4.5 is Active on Amazon Bedrock with no announced end-of-life date, and AWS gives at least 6 months of Legacy notice before end of life ([model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html), retrieved October 29, 2026)) |
| `embedding_model_id` | `amazon.titan-embed-text-v2:0` | lowest on-demand price of the Bedrock text embedding models compared, $0.02 per million tokens; retrieval-tuned; for visual/multimodal corpora evaluate Nova Multimodal Embeddings (requires new index + re-ingest) |
| `embedding_dimensions` | `1024` | 1024/512/256; immutable per vector index; 256 makes query data processing about 3.4x and vector storage about 1.9x cheaper in this pattern (see [Cost estimate](#cost-estimate-us-east-1-list-prices-retrieved-october-12-2026)); needs a new index (`vector_index_name`) and re-ingestion |
| `vector_index_name` | `chunks` | S3 Vectors index name; set a new name when you change dimensions or the embedding model |
| `enable_knowledge_base` | `false` | optional managed RAG path (semantic-only on S3 Vectors) |
| `api_throttle_rate` / `api_throttle_burst` | `10` / `20` | usage-plan throttling |
| `prefix` | `hybrid-rag` | resource naming |

Lambda-level knobs (env vars set by the compute stack): `MAX_CONTEXT_TOKENS`
(prompt context budget, default 6000), `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS`
(chunking, defaults 2048/256 - changes apply to newly ingested documents
only), `DEFAULT_TENANT`. Fusion weights and the candidate multiplier live in
`src/search/hybrid.py` (`dense_weight`, `sparse_weight`, `rrf_k`).

## Cost estimate (us-east-1 list prices retrieved October 12, 2026)

Scenario: 100k chunks (1,024-dim), 50k queries/month:

| Item | Monthly |
| --- | --- |
| S3 Vectors storage + queries | ≈ $0.25-3 (upper end includes one GetVectors call per query) |
| DynamoDB (BM25) + Lambda + Amazon API Gateway | ≈ $1.5-3 |
| **Retrieval path total** | **≈ $2-6** vs. about $175-350 (OpenSearch Serverless classic floor) / about $44 (Aurora Serverless v2 at 0.5 ACU, warm) |
| Bedrock tokens (Titan + Claude) | dominates at scale; identical across vector-store choices |

Observed in live testing: a typical `/query` uses ~1,800 input + ~100 output
tokens on Sonnet 4.6 - about $0.008 (under one cent) per question at the
Claude Sonnet 4.6 US cross-Region profile price ($3.30/$16.50 per million
input/output tokens).

Full worked comparison + decision tree: [docs/cost-comparison.md](docs/cost-comparison.md).
Migrating off OpenSearch: [docs/migration-guide.md](docs/migration-guide.md).

## Notable design points

- Chunk text lives in S3 Vectors **non-filterable metadata** - dense hits
  return passages without a second lookup; sparse-only hits hydrate via
  `GetVectors`.
- Tenant isolation is structural: metadata filter on the dense path,
  key-prefix partitioning on the sparse path. `tenant_id` must match
  `^[a-z0-9][a-z0-9_-]{0,63}$` after trimming and lowercasing; blank or
  omitted means the default tenant, and `dense_search()` never runs without a
  tenant filter. `tenant_id` comes from the payload (trusted-caller model);
  for untrusted clients, derive it from an authorizer (for example, an Amazon
  Cognito user pool claim or a tenant value that a Lambda authorizer returns
  in the request context) - a single function change, see
  docs/architecture.md.
- `maxTokens` is always explicit on Converse calls: when it's omitted it
  defaults to the model's maximum output, and Amazon Bedrock reserves input
  tokens plus `maxTokens` against your token quotas until the request
  finishes, a common cause of unexpected `ThrottlingException` errors.
  Prompts contain **no assistant prefill**: Claude Sonnet 4.6 and later
  models reject a prefilled final assistant turn with a 400 error.
- Responses are buffered JSON. Lambda supports response streaming only on
  Node.js managed runtimes (Python needs a custom runtime or the AWS Lambda
  Web Adapter), and API Gateway REST APIs buffer by default; their STREAM
  response transfer mode (since November 19, 2025, proxy integrations only)
  needs a function that streams. True token streaming needs a Function URL +
  `RESPONSE_STREAM` front (`stream_answer()` generator is ready for it) - see
  [docs/architecture.md](docs/architecture.md).
- Deletes propagate. Deleting a document from the bucket invokes the
  ingestion function through an `s3:ObjectRemoved` notification, which uses
  the document's manifest to delete its vectors and BM25 postings and
  decrement the tenant statistics (`document_deleted` log record). Unexpected
  ingestion errors are re-raised so that Lambda retries the asynchronous S3
  event (two more attempts by default); rejected uploads, unsupported
  formats, documents with no extractable text, and objects that no longer
  exist aren't retried. Ingestion is cleanup-first and manifest-driven, so a
  retry rewrites the same vector keys and postings, and a delete is a no-op
  once the manifest is gone. Each tenant statistics update is followed by a
  separate manifest write or delete (this applies to the cleanup step of a
  re-ingest, to the indexing step, and to a delete); if an attempt fails
  between the two, the retry applies the statistics change again, so
  `doc_count` and `total_len` over- or under-count by that document (BM25
  scoring only). If the object is overwritten between attempts by a version
  with fewer chunks, the extra chunks from the failed attempt stay: their
  postings until the document is next re-ingested or deleted, and their
  vectors (still returned by dense search) indefinitely. Events that fail
  every retry land in the ingestion function's on-failure destination, an
  Amazon SQS queue whose URL is the `IngestionFailureQueueUrl` stack output
  (messages are kept for 14 days).
- The compute stack exports the generated log group names as the
  `IngestionLogGroupName`, `SearchLogGroupName`, and `QueryLogGroupName`
  outputs, for example `aws logs tail <IngestionLogGroupName> --since 10m`.

## Testing

`pytest` runs 143 tests (unit + CDK assertions, no AWS credentials needed).

```bash
pytest                 # all: unit + CDK assertions (no AWS credentials needed)
pytest tests/unit      # chunker, BM25 math, RRF, embedder contract, prompt budget, handlers, tenant rule, ingestion events
pytest tests/infra     # synthesized-template assertions (bundling skipped)

# after deploying, end-to-end smoke test against the live stacks:
python scripts/smoke_test.py   # upload -> index -> search x3 -> grounded answer -> tenant isolation
```

## Cleanup

```bash
cd infra && cdk destroy --all   # add -c enable_knowledge_base=true if the KB stack was deployed
```

`cdk destroy` empties and deletes the document bucket and deletes the
DynamoDB table, the vector indexes, and the ingestion failure queue
(`IngestionFailureQueueUrl`), so you don't need to delete test documents
first. The account-level Amazon API Gateway CloudWatch Logs role that the
API stack sets (one per account and Region, shared by all REST APIs) is
retained so that other APIs in the account keep logging. To check delete propagation before you tear down, run
`python scripts/smoke_test.py --cleanup`. It reruns the full smoke test
(including one billed Claude `/query` call), then deletes its test document
and verifies that the document is no longer searchable.

The vector indexes are stack-managed and deleted in the right order
automatically. Don't delete them manually first: deleting them outside of AWS CloudFormation
leaves the stack out of sync with the deployed resources. If the
vector bucket ever refuses to delete because of an index created outside
these stacks:
`aws s3vectors delete-index --vector-bucket-name <name> --index-name <name>`,
then destroy again.

## Security

See [SECURITY.md](SECURITY.md) for the security policy, known trade-offs, and production hardening recommendations, and [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security issue.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
