"""Ingestion Lambda end to end against in-memory fakes: reject, retry, delete."""
import json
import logging
import urllib.parse

import pytest

from ingestion import handler as ingestion
from ingestion.bm25_indexer import manifest_key, stats_key

BUCKET = "docs-bucket"
DOC_TEXT = "S3 Vectors stores embeddings at low cost for retrieval workloads. " * 60


def _s3_event(key: str, event_name: str = "ObjectCreated:Put") -> dict:
    # S3 URL-encodes object keys in notifications (spaces become "+").
    encoded = urllib.parse.quote_plus(key, safe="/")
    return {
        "Records": [
            {
                "eventName": event_name,
                "s3": {"bucket": {"name": BUCKET}, "object": {"key": encoded}},
            }
        ]
    }


def _logged(caplog) -> list[dict]:
    return [
        json.loads(r.getMessage()) for r in caplog.records if r.name == ingestion.logger.name
    ]


def _events(caplog) -> list[str]:
    return [entry["message"] for entry in _logged(caplog)]


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO, logger=ingestion.logger.name)


def test_invalid_tenant_folder_is_rejected_not_ingested(ingestion_env, caplog):
    key = "tenants/Anyexamplecompany Corp/x.txt"
    ingestion_env.s3.put_object(Bucket=BUCKET, Key=key, Body=DOC_TEXT)
    ingestion_env.s3.calls.clear()

    result = ingestion.lambda_handler(_s3_event(key), None)  # must not raise

    assert result["results"][0]["status"] == "rejected"
    assert "invalid tenant folder" in result["results"][0]["reason"]
    assert "get_object" not in ingestion_env.s3.call_names()
    assert ingestion_env.s3vectors.calls == []
    assert ingestion_env.table.items == {}
    rejected = [e for e in _logged(caplog) if e["message"] == "document_rejected"]
    assert rejected and rejected[0]["key"] == key and "reason" in rejected[0]


def test_unexpected_error_propagates_for_lambda_retry(ingestion_env, caplog):
    key = "tenants/anyexamplecompany/notes.txt"
    ingestion_env.s3.put_object(Bucket=BUCKET, Key=key, Body=DOC_TEXT)
    ingestion_env.s3vectors.fail_on["put_vectors"] = RuntimeError("throttled")

    with pytest.raises(RuntimeError, match="throttled"):
        ingestion.lambda_handler(_s3_event(key), None)

    failed = [e for e in _logged(caplog) if e["message"] == "ingestion_failed"]
    assert failed and failed[0]["key"] == key and failed[0]["error"] == "throttled"


def test_validation_outcomes_are_skipped_without_raising(ingestion_env):
    ingestion_env.s3.put_object(Bucket=BUCKET, Key="tenants/anyexamplecompany/blank.txt", Body="   \n ")

    outcomes = {
        key: ingestion.lambda_handler(_s3_event(key), None)["results"][0]
        for key in ("tenants/anyexamplecompany/x.exe", "tenants/anyexamplecompany/blank.txt", "tenants/anyexamplecompany/gone.txt")
    }

    assert {k: v["status"] for k, v in outcomes.items()} == {
        "tenants/anyexamplecompany/x.exe": "skipped",
        "tenants/anyexamplecompany/blank.txt": "skipped",
        "tenants/anyexamplecompany/gone.txt": "skipped",
    }
    assert outcomes["tenants/anyexamplecompany/x.exe"]["reason"] == "unsupported format"
    assert outcomes["tenants/anyexamplecompany/blank.txt"]["reason"] == "no extractable text"
    assert outcomes["tenants/anyexamplecompany/gone.txt"]["reason"] == "object no longer exists"
    assert ingestion_env.s3vectors.calls == []


def test_delete_event_removes_vectors_postings_stats_and_manifest(ingestion_env, caplog):
    key = "tenants/anyexamplecompany/notes.txt"
    ingestion_env.s3.put_object(Bucket=BUCKET, Key=key, Body=DOC_TEXT)
    ingested = ingestion.lambda_handler(_s3_event(key), None)["results"][0]
    assert ingested["status"] == "ingested" and ingested["chunks"] >= 2

    ingested_keys = [
        v["key"] for call in ingestion_env.s3vectors.calls_to("put_vectors") for v in call["vectors"]
    ]
    doc_id = ingested["doc_id"]
    assert any("gsi1pk" in item for item in ingestion_env.table.items.values())

    ingestion_env.s3.delete_object(Bucket=BUCKET, Key=key)
    s3_calls_before = len(ingestion_env.s3.calls)

    deleted = ingestion.lambda_handler(_s3_event(key, "ObjectRemoved:Delete"), None)["results"][0]

    assert deleted["status"] == "deleted"
    assert deleted["chunks_removed"] == len(ingested_keys)
    removed_keys = [k for call in ingestion_env.s3vectors.calls_to("delete_vectors") for k in call["keys"]]
    assert removed_keys == ingested_keys

    items = ingestion_env.table.items
    assert not any("gsi1pk" in item for item in items.values())  # no postings left
    assert (manifest_key("anyexamplecompany", doc_id)["PK"], "MANIFEST") not in items
    stats = items[(stats_key("anyexamplecompany")["PK"], "GLOBAL")]
    assert stats["doc_count"] == 0 and stats["total_len"] == 0

    assert "get_object" not in [name for name, _ in ingestion_env.s3.calls[s3_calls_before:]]
    assert "document_deleted" in _events(caplog)


def test_delete_event_for_never_ingested_document_is_a_no_op(ingestion_env):
    result = ingestion.lambda_handler(
        _s3_event("tenants/anyexamplecompany/never.txt", "ObjectRemoved:Delete"), None
    )["results"][0]

    assert result["status"] == "not_indexed"
    assert ingestion_env.s3vectors.calls_to("delete_vectors") == []


def test_late_delete_event_does_not_wipe_a_reupload(ingestion_env):
    key = "tenants/anyexamplecompany/notes.txt"
    ingestion_env.s3.put_object(Bucket=BUCKET, Key=key, Body=DOC_TEXT)
    ingestion.lambda_handler(_s3_event(key), None)
    items_before = dict(ingestion_env.table.items)

    # The object still exists (re-uploaded after the delete that fired this event).
    result = ingestion.lambda_handler(_s3_event(key, "ObjectRemoved:Delete"), None)["results"][0]

    assert result["status"] == "skipped" and result["reason"] == "object exists again"
    assert ingestion_env.s3vectors.calls_to("delete_vectors") == []
    assert ingestion_env.table.items == items_before
