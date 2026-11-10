"""Dense retrieval path: S3 Vectors approximate nearest neighbor search.

Verified API notes:
  * ``query_vectors``: queryVector={"float32": [...]}, required topK, optional
    metadata ``filter``, returnMetadata/returnDistance flags.
  * Filter timing depends on the index mode. ENHANCED (the default for vector
    buckets created on or after 2026-09-30) applies the filter before the
    vector search; CLASSIC filters during the search and can return fewer
    than top-k results.
  * The tenant filter is ALWAYS sent: a query without a tenant is refused
    rather than searching every tenant's vectors.
  * topK is capped at 100, the QueryVectors page size. boto3 1.40.0 (pinned
    in src/requirements.txt) has no QueryVectors ``nextToken`` support, so
    results beyond the first page can't be fetched.
  * The index uses cosine distance: lower distance = more similar.
  * ``get_vectors`` hydrates metadata for up to 100 keys per call - used for
    sparse-only hits whose text we did not receive from the dense query.
"""
from common import aws_clients, config

# Results per QueryVectors response page. Requests above this would silently
# return only the first page with the pinned SDK, so cap topK instead.
QUERY_VECTORS_MAX_RESULTS = 100


def dense_search(query_embedding: list[float], tenant: str, top_k: int = 10) -> list[dict]:
    """Return ANN results for one tenant: [{key, distance, metadata}] by similarity."""
    if not tenant or not tenant.strip():
        raise ValueError("dense_search requires a tenant: every vector query is scoped by tenant_id")

    response = aws_clients.s3vectors().query_vectors(
        vectorBucketName=config.vector_bucket_name(),
        indexName=config.vector_index_name(),
        queryVector={"float32": [float(v) for v in query_embedding]},
        topK=min(int(top_k), QUERY_VECTORS_MAX_RESULTS),
        filter={"tenant_id": {"$eq": tenant}},
        returnMetadata=True,
        returnDistance=True,
    )
    return [
        {
            "key": v["key"],
            "distance": v.get("distance"),
            "metadata": v.get("metadata", {}) or {},
        }
        for v in response.get("vectors", [])
    ]


def fetch_vectors_metadata(keys: list[str]) -> dict[str, dict]:
    """Hydrate metadata (including chunk text) for arbitrary vector keys."""
    if not keys:
        return {}
    client = aws_clients.s3vectors()
    result: dict[str, dict] = {}
    for start in range(0, len(keys), 100):  # GetVectors max 100 keys/call
        response = client.get_vectors(
            vectorBucketName=config.vector_bucket_name(),
            indexName=config.vector_index_name(),
            keys=keys[start : start + 100],
            returnMetadata=True,
            returnData=False,
        )
        for v in response.get("vectors", []):
            result[v["key"]] = v.get("metadata", {}) or {}
    return result
