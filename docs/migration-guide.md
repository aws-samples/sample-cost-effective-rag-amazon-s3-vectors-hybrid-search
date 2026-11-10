# Migrating from Amazon OpenSearch Service vector stores to Amazon S3 Vectors

A practical path from an Amazon OpenSearch Service or
Amazon OpenSearch Serverless k-NN index to the S3 Vectors architecture in
this pattern.

## Should you migrate? (read this first)

Migrate when your OpenSearch cluster mostly serves *vector* similarity for a
RAG workload with modest QPS - you are paying a compute floor for what is
effectively storage plus occasional math. OpenSearch Serverless classic
collections bill at least about $175/month (dev-test) or $350/month (HA) even
when idle; NextGen collections scale to zero with a 10-30 second cold start
(us-east-1, retrieved October 12, 2026). See `docs/cost-comparison.md` for
the break-even analysis.

Do **not** migrate (or migrate only a tier) when you rely on OpenSearch
features S3 Vectors does not have:

| OpenSearch feature | S3 Vectors equivalent |
| --- | --- |
| Native BM25 / hybrid queries | This pattern's Amazon DynamoDB BM25 + RRF |
| Aggregations, faceting, analyzers | None - keep OpenSearch for these |
| Single-digit-ms p99 | None - S3 Vectors delivers subsecond latency for infrequent queries and as low as 100 ms for more frequent queries ([Working with S3 Vectors and vector buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)) |
| Index-time filters on any field | Filterable metadata (up to 2 KB per vector; every metadata key is filterable unless declared non-filterable at index creation) |
| Score scripting / custom similarity | cosine or euclidean only |

## Concept mapping

| OpenSearch | S3 Vectors |
| --- | --- |
| Domain / collection | Vector bucket |
| Index (k-NN enabled) | Vector index (dimension + distance metric fixed at creation) |
| Document `_id` | Vector `key` |
| `knn_vector` field | Vector `data` (float32) |
| Document fields | Vector `metadata` (filterable, ≤ 2 KB; 40 KB total, 50 keys) |
| `_source` large fields | Non-filterable metadata (declared at index creation, ≤ 10 keys) |
| k-NN query + `filter` | `QueryVectors` with metadata `filter`: ENHANCED (default for vector buckets created on or after September 30, 2026) pre-filters; CLASSIC filters during the search and can return fewer than top-k |
| Bulk API | `PutVectors` (≤ 500 vectors / call, ≤ 20 MiB payload) |

## Migration steps

### 1. Deploy the target (this pattern)

```bash
cd infra && cdk deploy --all
```

Create the vector index with the **same dimension** as your existing
embeddings and the distance metric your embedding model recommends (cosine
for Amazon Titan Text Embeddings V2). Dimensions are immutable per index -
if you also plan to switch embedding models, re-embedding (step 3b) is the moment.

### 2. Export from OpenSearch

Scroll the source index and dump `_id`, the vector field, and the metadata
fields you will keep (Python sketch, `opensearch-py`):

```python
from opensearchpy import OpenSearch, helpers

client = OpenSearch(hosts=[SOURCE_HOST], ...)
for hit in helpers.scan(client, index="rag-chunks", query={"query": {"match_all": {}}}):
    src = hit["_source"]
    yield {
        "key": hit["_id"],
        "vector": src["embedding"],          # your knn_vector field
        "text": src["chunk_text"],
        "metadata": {
            "tenant_id": src.get("tenant", "default"),
            "doc_id": src["doc_id"],
            "source": src["source"],
        },
    }
```

### 3. Transform

- **a. Keys** - S3 Vectors keys are strings unique per index. This pattern's
  convention: `<tenant>#<doc_id>#<chunk_index:05d>` (enables derived-key
  cleanup). Map your `_id`s or regenerate.
- **b. Vectors** - reuse exported embeddings as float32 if you keep the same
  model. Changing models (e.g. to Titan V2 1024-d) means re-embedding the
  chunk text instead - skip exporting vectors and run this pattern's
  ingestion over the raw documents, which also builds the BM25 index for you.
- **c. Metadata** - keep filterable metadata under 2 KB/vector: move long
  text into the non-filterable `text` key; drop fields you never filter on.
  Tenant IDs must match `^[a-z0-9][a-z0-9_-]{0,63}$` (lowercase), because
  `/search` and `/query` reject other values.

### 4. Load into S3 Vectors

```python
import boto3

s3v = boto3.client("s3vectors")
BATCH = 500  # API maximum per PutVectors call

def load(batch):
    s3v.put_vectors(
        vectorBucketName=BUCKET,
        indexName="chunks",
        vectors=[
            {
                "key": r["key"],
                "data": {"float32": [float(x) for x in r["vector"]]},
                "metadata": {**r["metadata"], "text": r["text"]},
            }
            for r in batch
        ],
    )
```

Throughput planning
([S3 Vectors limitations](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html),
retrieved October 12, 2026): ≤ 1,000 combined write requests/s and ≤ 2,500
vectors written/s per index - a 10M-vector corpus loads in a bit over an hour
at the limit. Large batches also avoid paying the 128 KB minimum billed per
PutVectors call ($0.20 per GB uploaded). Use adaptive retries; expect
occasional `TooManyRequestsException`.

### 5. Build the sparse (BM25) index

If you loaded pre-embedded vectors directly (4), the DynamoDB BM25 index is
still empty. Either:

- re-run this pattern's ingestion AWS Lambda function over the original documents (it
  builds both indexes and the manifests), or
- call `bm25_indexer.index_chunks()` over your exported `(key, text)` pairs
  in a one-off script - it is the same code path the Lambda uses.

Vectors loaded directly have no manifest, so deleting their source object
doesn't remove them; use the ingestion function path if deletes must
propagate.

### 6. Validate before cutover

- **Recall parity**: for a golden set of queries, compare top-k overlap
  between OpenSearch and the new `/search` endpoint (modes `dense` and
  `hybrid`). Expect small ANN differences; investigate overlaps below ~80%.
- **Latency**: measure p50/p95 on `/search` against your OpenSearch baseline
  and confirm it is acceptable for your UX. S3 Vectors queries alone are
  subsecond, as low as 100 ms for more frequent queries
  ([Working with S3 Vectors and vector buckets](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)); `/search` adds the
  embedding call and the BM25 lookup.
- **Filters**: verify tenant filters return only tenant data.
- **Dual-read window**: run both stores behind a flag, log result diffs on
  real traffic before switching writes.

### 7. Cut over and decommission

1. Point ingestion at the S3 document bucket (writes now flow to S3 Vectors + DynamoDB).
2. Flip readers to the new `/search` / `/query` endpoints.
3. Keep the OpenSearch snapshot for the rollback window (a final snapshot to
   S3 is cheap insurance).
4. Delete the OpenSearch domain/collection - this is where the savings start.

### Rollback plan

If you already deleted the domain, restore it from the final snapshot; test
the restore during the rollback window. Because step 6 ran dual-read, the old cluster remains warm until you delete it; rollback is
flipping the reader flag back.
