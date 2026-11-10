import json

import search.hybrid as hybrid_mod
from search.hybrid import hybrid_search


def _patch_retrieval(monkeypatch, dense, sparse, hydrated=None):
    """Stub the retrieval seams inside search.hybrid."""
    monkeypatch.setattr(hybrid_mod, "embed_text", lambda q: [0.1] * 8)
    monkeypatch.setattr(hybrid_mod, "dense_search", lambda emb, tenant, top_k: dense)
    monkeypatch.setattr(hybrid_mod, "bm25_search", lambda q, tenant, top_k: sparse)
    monkeypatch.setattr(hybrid_mod, "fetch_vectors_metadata", lambda keys: hydrated or {})


def test_hybrid_fuses_and_hydrates_sparse_only_hits(monkeypatch):
    dense = [
        {"key": "k1", "distance": 0.10, "metadata": {"text": "dense one", "doc_id": "d1", "source": "s1"}},
        {"key": "k2", "distance": 0.20, "metadata": {"text": "dense two", "doc_id": "d2", "source": "s2"}},
    ]
    sparse = [{"key": "k3", "score": 9.5}, {"key": "k1", "score": 4.0}]
    hydrated = {"k3": {"text": "sparse only text", "doc_id": "d3", "source": "s3"}}
    _patch_retrieval(monkeypatch, dense, sparse, hydrated)

    results = hybrid_search("query", "default", top_k=3, mode="hybrid")

    assert results[0]["key"] == "k1"  # appears in both lists -> highest RRF
    by_key = {r["key"]: r for r in results}
    assert by_key["k3"]["text"] == "sparse only text"  # hydrated via GetVectors
    assert by_key["k3"]["dense_rank"] is None and by_key["k3"]["sparse_rank"] == 1
    assert by_key["k1"]["dense_rank"] == 1 and by_key["k1"]["sparse_rank"] == 2
    assert "text" not in by_key["k1"]["metadata"]  # text lifted out of metadata


def test_dense_only_and_sparse_only_modes(monkeypatch):
    dense = [{"key": "k1", "distance": 0.25, "metadata": {"text": "t", "doc_id": "d"}}]
    sparse = [{"key": "k9", "score": 3.3}]
    _patch_retrieval(monkeypatch, dense, sparse, {"k9": {"text": "s"}})

    dense_results = hybrid_search("q", "default", top_k=5, mode="dense")
    assert [r["key"] for r in dense_results] == ["k1"]
    assert dense_results[0]["score"] == 0.75  # 1 - cosine distance

    sparse_results = hybrid_search("q", "default", top_k=5, mode="sparse")
    assert [r["key"] for r in sparse_results] == ["k9"]
    assert sparse_results[0]["score"] == 3.3


def test_query_handler_happy_path(monkeypatch, fake_bedrock):
    import query.handler as qh

    retrieved = [
        {
            "key": "default#doc-1#00000",
            "score": 0.032,
            "text": "S3 Vectors reduces storage cost by up to 90 percent.",
            "metadata": {"doc_id": "doc-1", "source": "docs/pricing.md"},
            "dense_rank": 1,
            "sparse_rank": 2,
        }
    ]
    monkeypatch.setattr(qh, "hybrid_search", lambda *a, **k: retrieved)

    event = {"body": json.dumps({"question": "How much cheaper is S3 Vectors?", "top_k": 3})}
    response = qh.lambda_handler(event, None)

    assert response["statusCode"] == 200
    payload = json.loads(response["body"])
    assert payload["answer"] == fake_bedrock.converse_reply
    assert payload["citations"][0]["source"] == "docs/pricing.md"
    assert payload["usage"]["outputTokens"] == 20

    call = fake_bedrock.converse_calls[-1]
    assert call["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert call["inferenceConfig"]["maxTokens"] == 1024  # always explicit
    assert [m["role"] for m in call["messages"]] == ["user"]  # no prefill
    assert "cite" in call["system"][0]["text"].lower()


def test_query_handler_no_results_short_circuits_llm(monkeypatch, fake_bedrock):
    import query.handler as qh

    monkeypatch.setattr(qh, "hybrid_search", lambda *a, **k: [])
    response = qh.lambda_handler({"body": json.dumps({"question": "anything?"})}, None)

    assert response["statusCode"] == 200
    assert "could not find" in json.loads(response["body"])["answer"]
    assert fake_bedrock.converse_calls == []  # no wasted generation spend


def test_query_handler_explicit_zero_temperature_is_respected(monkeypatch, fake_bedrock):
    """Regression: `temperature or 0.2` silently replaced a valid explicit 0.0."""
    import query.handler as qh

    retrieved = [{"key": "k", "score": 1.0, "text": "ctx", "metadata": {}, "dense_rank": 1, "sparse_rank": None}]
    monkeypatch.setattr(qh, "hybrid_search", lambda *a, **k: retrieved)

    event = {"body": json.dumps({"question": "q?", "temperature": 0})}
    assert qh.lambda_handler(event, None)["statusCode"] == 200
    assert fake_bedrock.converse_calls[-1]["inferenceConfig"]["temperature"] == 0.0

    # Omitting it still applies the default.
    qh.lambda_handler({"body": json.dumps({"question": "q?"})}, None)
    assert fake_bedrock.converse_calls[-1]["inferenceConfig"]["temperature"] == 0.2


def test_query_handler_validation():
    import query.handler as qh

    assert qh.lambda_handler({"body": "not json"}, None)["statusCode"] == 400
    assert qh.lambda_handler({"body": json.dumps({})}, None)["statusCode"] == 400


def test_search_handler_happy_path_and_validation(monkeypatch):
    import search.handler as sh

    monkeypatch.setattr(
        sh, "hybrid_search", lambda q, tenant, top_k, mode: [{"key": "k", "score": 1.0, "text": "t"}]
    )
    ok = sh.lambda_handler(
        {"body": json.dumps({"query": "hello", "tenant_id": "Anyexamplecompany", "mode": "hybrid"})}, None
    )
    assert ok["statusCode"] == 200
    payload = json.loads(ok["body"])
    assert payload["tenant_id"] == "anyexamplecompany"  # normalized
    assert payload["results"][0]["key"] == "k"

    assert sh.lambda_handler({"body": json.dumps({})}, None)["statusCode"] == 400
    bad_mode = sh.lambda_handler({"body": json.dumps({"query": "x", "mode": "fuzzy"})}, None)
    assert bad_mode["statusCode"] == 400
