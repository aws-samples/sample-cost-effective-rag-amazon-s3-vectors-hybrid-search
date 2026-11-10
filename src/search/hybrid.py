"""Hybrid search orchestrator: dense ANN + sparse BM25 -> RRF -> hydrated results.

Used by BOTH the /search Lambda handler and (in-process) the /query Lambda,
avoiding a Lambda-to-Lambda hop on the answer path.
"""
from common.logging_util import get_logger, log_json
from ingestion.embedder import embed_text
from search.dense_search import dense_search, fetch_vectors_metadata
from search.rrf_fusion import DEFAULT_RRF_K, rrf_fuse
from search.sparse_search import bm25_search

logger = get_logger(__name__)

VALID_MODES = ("hybrid", "dense", "sparse")


def hybrid_search(
    query: str,
    tenant: str,
    top_k: int = 5,
    mode: str = "hybrid",
    *,
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    candidate_multiplier: int = 3,
) -> list[dict]:
    """Run retrieval and return hydrated results, best first.

    Each result: {key, score, text, metadata, dense_rank, sparse_rank}.
    ``score`` is the RRF score in hybrid mode; the native score otherwise.
    In hybrid mode, ties in the RRF score go to the better sparse rank, then
    the better dense rank (see ``search.rrf_fusion``).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    # Retrieve more candidates than requested so fusion has signal to work with.
    candidate_k = max(top_k * candidate_multiplier, 10)

    dense_results: list[dict] = []
    sparse_results: list[dict] = []
    if mode in ("hybrid", "dense"):
        dense_results = dense_search(embed_text(query), tenant, top_k=candidate_k)
    if mode in ("hybrid", "sparse"):
        sparse_results = bm25_search(query, tenant, top_k=candidate_k)

    dense_keys = [r["key"] for r in dense_results]
    sparse_keys = [r["key"] for r in sparse_results]

    if mode == "hybrid":
        fused = rrf_fuse(
            {"dense": dense_keys, "sparse": sparse_keys},
            k=rrf_k,
            weights={"dense": dense_weight, "sparse": sparse_weight},
        )
    elif mode == "dense":
        fused = [
            {"key": k, "score": 1.0 - (r.get("distance") or 0.0), "ranks": {"dense": i}}
            for i, (k, r) in enumerate(zip(dense_keys, dense_results), start=1)
        ]
    else:  # sparse
        fused = [
            {"key": r["key"], "score": r["score"], "ranks": {"sparse": i}}
            for i, r in enumerate(sparse_results, start=1)
        ]

    top = fused[:top_k]

    # Hydrate text/metadata: dense hits already carry metadata; fetch the rest.
    metadata_by_key = {r["key"]: r["metadata"] for r in dense_results}
    missing = [item["key"] for item in top if item["key"] not in metadata_by_key]
    metadata_by_key.update(fetch_vectors_metadata(missing))

    results = []
    for item in top:
        meta = dict(metadata_by_key.get(item["key"], {}))
        text = meta.pop("text", "")
        results.append(
            {
                "key": item["key"],
                "score": item["score"],
                "text": text,
                "metadata": meta,
                "dense_rank": item["ranks"].get("dense"),
                "sparse_rank": item["ranks"].get("sparse"),
            }
        )

    log_json(
        logger,
        "hybrid_search",
        tenant=tenant,
        mode=mode,
        dense_candidates=len(dense_results),
        sparse_candidates=len(sparse_results),
        returned=len(results),
    )
    return results
