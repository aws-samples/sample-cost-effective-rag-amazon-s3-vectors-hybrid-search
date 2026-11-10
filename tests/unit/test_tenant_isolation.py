"""Tenant scoping: request normalization, the dense filter, and the 100-result cap."""
import json

import pytest

from common import aws_clients
from common.tenant import is_valid_tenant_id, resolve_request_tenant
from search.dense_search import QUERY_VECTORS_MAX_RESULTS, dense_search

# --- shared tenant rule -----------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_blank_or_missing_tenant_resolves_to_default(raw):
    assert resolve_request_tenant(raw) == "default"


def test_tenant_is_trimmed_and_lowercased():
    assert resolve_request_tenant(" Anyexamplecompany ") == "anyexamplecompany"
    assert resolve_request_tenant("x" * 64) == "x" * 64


@pytest.mark.parametrize("raw", ["a b", "-anyexamplecompany", "_x", "anyexamplecompany!", "x" * 65, 123])
def test_invalid_tenant_raises(raw):
    with pytest.raises(ValueError):
        resolve_request_tenant(raw)


def test_is_valid_tenant_id_uses_full_match():
    assert is_valid_tenant_id("anyexamplecompany")
    assert not is_valid_tenant_id("anyexamplecompany\n")  # re.match + "$" would accept this
    assert not is_valid_tenant_id("Anyexamplecompany")  # not normalized
    assert not is_valid_tenant_id(None)


# --- dense_search -------------------------------------------------------------


@pytest.fixture
def s3vectors(monkeypatch, fake_s3vectors):
    monkeypatch.setattr(aws_clients, "s3vectors", lambda: fake_s3vectors)
    return fake_s3vectors


def test_dense_search_always_filters_by_tenant(s3vectors):
    dense_search([0.1] * 8, "anyexamplecompany", top_k=5)
    (call,) = s3vectors.calls_to("query_vectors")
    assert call["filter"] == {"tenant_id": {"$eq": "anyexamplecompany"}}


@pytest.mark.parametrize("tenant", [None, "", "   "])
def test_dense_search_refuses_unscoped_queries(s3vectors, tenant):
    with pytest.raises(ValueError):
        dense_search([0.1] * 8, tenant, top_k=5)
    assert s3vectors.calls == []


def test_dense_search_caps_top_k_at_query_vectors_page_size(s3vectors):
    dense_search([0.1] * 8, "anyexamplecompany", top_k=150)
    dense_search([0.1] * 8, "anyexamplecompany", top_k=30)
    assert QUERY_VECTORS_MAX_RESULTS == 100
    assert [c["topK"] for c in s3vectors.calls_to("query_vectors")] == [100, 30]


# --- handlers -----------------------------------------------------------------


def _capture_hybrid(monkeypatch, module):
    seen = []

    def fake_hybrid(query, tenant, **kwargs):
        seen.append(tenant)
        return []

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    return seen


def test_search_handler_blank_tenant_uses_default(monkeypatch):
    import search.handler as sh

    seen = _capture_hybrid(monkeypatch, sh)
    response = sh.lambda_handler({"body": json.dumps({"query": "hello", "tenant_id": "   "})}, None)

    assert response["statusCode"] == 200
    assert seen == ["default"]
    assert json.loads(response["body"])["tenant_id"] == "default"


def test_query_handler_blank_tenant_uses_default(monkeypatch, fake_bedrock):
    import query.handler as qh

    seen = _capture_hybrid(monkeypatch, qh)
    response = qh.lambda_handler({"body": json.dumps({"question": "q?", "tenant_id": " "})}, None)

    assert response["statusCode"] == 200
    assert seen == ["default"]


def test_handlers_reject_invalid_tenant_without_searching(monkeypatch, fake_bedrock):
    import query.handler as qh
    import search.handler as sh

    search_seen = _capture_hybrid(monkeypatch, sh)
    query_seen = _capture_hybrid(monkeypatch, qh)

    search_resp = sh.lambda_handler(
        {"body": json.dumps({"query": "hello", "tenant_id": "bad tenant!"})}, None
    )
    query_resp = qh.lambda_handler(
        {"body": json.dumps({"question": "q?", "tenant_id": "bad tenant!"})}, None
    )

    assert search_resp["statusCode"] == 400 and query_resp["statusCode"] == 400
    assert "tenant_id" in json.loads(search_resp["body"])["error"]
    assert search_seen == [] and query_seen == []


def test_blank_tenant_end_to_end_sends_default_tenant_filter(monkeypatch, s3vectors):
    """Request -> handler -> real hybrid_search -> real dense_search -> QueryVectors."""
    import search.handler as sh
    import search.hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "embed_text", lambda q: [0.1] * 8)
    monkeypatch.setattr(hybrid_mod, "bm25_search", lambda q, tenant, top_k: [])

    response = sh.lambda_handler(
        {"body": json.dumps({"query": "hello", "tenant_id": "   ", "mode": "hybrid"})}, None
    )

    assert response["statusCode"] == 200
    (call,) = s3vectors.calls_to("query_vectors")
    assert call["filter"] == {"tenant_id": {"$eq": "default"}}
