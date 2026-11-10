"""Ingestion Lambda: S3 ObjectCreated / ObjectRemoved -> keep both indexes in sync.

Pipeline per uploaded document (ObjectCreated):
    1. Extract text (PDF / TXT / MD / HTML).
    2. Recursive character chunking (~512 tokens, 12.5% overlap).
    3. Titan Text Embeddings V2 per chunk.
    4. PutVectors into the S3 Vectors index - filterable metadata
       (tenant_id, doc_id, source, chunk_index, format, ingested_at) plus the
       chunk text as non-filterable metadata.
    5. BM25 postings + tenant stats into DynamoDB.
    6. Manifest item enables precise cleanup on re-ingestion and deletion.

Deleted document (ObjectRemoved): the manifest-driven cleanup used for
re-ingestion removes its vectors and BM25 postings, decrements the tenant
stats, and deletes the manifest.

Multi-tenancy: keys following ``tenants/<tenant_id>/...`` are ingested under
that tenant. The folder name is lowercased and must match
``^[a-z0-9][a-z0-9_-]{0,63}$`` (common/tenant.py); objects under a folder
that doesn't are rejected (``document_rejected`` log, nothing indexed).
Any other key, including ``tenants/<file>``, lands in the default tenant.
"""
import hashlib
import re
import time
import urllib.parse

from botocore.exceptions import ClientError

from common import aws_clients, config
from common.logging_util import get_logger, log_json
from common.tenant import is_valid_tenant_id
from ingestion import bm25_indexer, embedder, parsers
from ingestion.chunker import chunk_text

logger = get_logger(__name__)

_PUT_VECTORS_BATCH = 100  # API max is 500; 100 keeps payloads comfortably small
_DELETE_VECTORS_BATCH = 500
_NOT_FOUND_CODES = ("NoSuchKey", "404")  # GetObject / HeadObject on a missing key


def tenant_from_key(key: str) -> str | None:
    """``tenants/<tenant>/...`` -> lowercased tenant id; other keys -> default tenant.

    Returns None when the key is under ``tenants/<folder>/`` but the folder
    isn't a valid tenant ID. Callers reject such objects: falling back to the
    default tenant would expose them to every caller that omits tenant_id.
    """
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "tenants":
        candidate = parts[1].lower()
        return candidate if is_valid_tenant_id(candidate) else None
    return config.default_tenant()


def doc_id_from_key(key: str) -> str:
    """Stable, readable document id: <basename-slug>-<md5(key)[:8]>.

    MD5 is used purely as a fast, stable identifier hash - not for any
    security purpose (hence usedforsecurity=False).
    """
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    basename = key.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", basename).strip("-")[:40] or "doc"
    return f"{slug}-{digest}"


def vector_key(tenant: str, doc_id: str, chunk_index: int) -> str:
    return f"{tenant}#{doc_id}#{chunk_index:05d}"


def _is_not_found(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES


def _skipped(key: str, reason: str) -> dict:
    summary = {"key": key, "status": "skipped", "reason": reason}
    log_json(logger, "document_skipped", **summary)
    return summary


def _rejected(key: str) -> dict:
    folder = key.split("/")[1]
    summary = {
        "key": key,
        "status": "rejected",
        "reason": (
            f"invalid tenant folder: {folder!r} (lowercased, it must match "
            "^[a-z0-9][a-z0-9_-]{0,63}$)"
        ),
    }
    log_json(logger, "document_rejected", level=30, **summary)
    return summary


def _delete_indexed_document(tenant: str, doc_id: str, table) -> int | None:
    """Remove a document's vectors, postings, stats contribution, and manifest.

    Manifest-driven. Returns the chunk count the manifest recorded, or None
    if the document has no manifest (never indexed, or already removed).
    """
    manifest = table.get_item(Key=bm25_indexer.manifest_key(tenant, doc_id)).get("Item")
    if not manifest:
        return None

    old_count = int(manifest.get("chunk_count", 0))
    old_len = int(manifest.get("total_len", 0))

    keys = [vector_key(tenant, doc_id, i) for i in range(old_count)]
    client = aws_clients.s3vectors()
    for start in range(0, len(keys), _DELETE_VECTORS_BATCH):
        client.delete_vectors(
            vectorBucketName=config.vector_bucket_name(),
            indexName=config.vector_index_name(),
            keys=keys[start : start + _DELETE_VECTORS_BATCH],
        )

    bm25_indexer.delete_document_postings(tenant, doc_id, table=table)
    bm25_indexer.decrement_stats(tenant, old_count, old_len, table=table)
    table.delete_item(Key=bm25_indexer.manifest_key(tenant, doc_id))
    return old_count


def ingest_object(bucket: str, key: str) -> dict:
    """Ingest one S3 object; returns a summary dict."""
    fmt = parsers.format_from_key(key)
    if fmt is None:
        return _skipped(key, "unsupported format")

    tenant = tenant_from_key(key)
    if tenant is None:
        return _rejected(key)

    try:
        raw = aws_clients.s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if _is_not_found(exc):  # deleted or replaced before we ran; retrying can't help
            return _skipped(key, "object no longer exists")
        raise
    text = parsers.extract_text(raw, fmt)
    if not text.strip():
        return _skipped(key, "no extractable text")

    doc_id = doc_id_from_key(key)
    table = aws_clients.dynamodb_table(config.bm25_table_name())

    old_chunks = _delete_indexed_document(tenant, doc_id, table)
    if old_chunks is not None:
        log_json(logger, "reingest_cleanup", tenant=tenant, doc_id=doc_id, old_chunks=old_chunks)

    chunks = chunk_text(
        text,
        chunk_chars=config.chunk_chars(),
        overlap_chars=config.chunk_overlap_chars(),
    )
    now = int(time.time())
    chunk_keys = [vector_key(tenant, doc_id, i) for i in range(len(chunks))]

    # --- dense index: embed and write vectors in batches -------------------
    client = aws_clients.s3vectors()
    for start in range(0, len(chunks), _PUT_VECTORS_BATCH):
        batch_chunks = chunks[start : start + _PUT_VECTORS_BATCH]
        batch_keys = chunk_keys[start : start + _PUT_VECTORS_BATCH]
        embeddings = embedder.embed_texts(batch_chunks)
        client.put_vectors(
            vectorBucketName=config.vector_bucket_name(),
            indexName=config.vector_index_name(),
            vectors=[
                {
                    "key": k,
                    "data": {"float32": emb},
                    "metadata": {
                        # filterable (2 KB limit per vector):
                        "tenant_id": tenant,
                        "doc_id": doc_id,
                        "source": key,
                        "chunk_index": start + offset,
                        "format": fmt,
                        "ingested_at": now,
                        # non-filterable (declared at index creation):
                        "text": chunk,
                    },
                }
                for offset, (k, chunk, emb) in enumerate(zip(batch_keys, batch_chunks, embeddings))
            ],
        )

    # --- sparse index: BM25 postings + stats + manifest ---------------------
    total_len = bm25_indexer.index_chunks(tenant, doc_id, chunk_keys, chunks, table=table)
    table.put_item(
        Item={
            **bm25_indexer.manifest_key(tenant, doc_id),
            "chunk_count": len(chunks),
            "total_len": int(total_len),
            "s3_key": key,
            "ingested_at": now,
        }
    )

    summary = {
        "key": key,
        "status": "ingested",
        "tenant": tenant,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "format": fmt,
    }
    log_json(logger, "document_ingested", **summary)
    return summary


def remove_object(bucket: str, key: str) -> dict:
    """Handle a deleted S3 object: drop its vectors, postings, stats, manifest."""
    if parsers.format_from_key(key) is None:
        return _skipped(key, "unsupported format")

    tenant = tenant_from_key(key)
    if tenant is None:
        return _rejected(key)

    # S3 doesn't guarantee event order. If the key exists again, it was
    # re-uploaded after this delete and the newer upload owns the index
    # entries, so a late delete event must not wipe them.
    try:
        aws_clients.s3().head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if not _is_not_found(exc):
            raise
    else:
        return _skipped(key, "object exists again")

    doc_id = doc_id_from_key(key)
    table = aws_clients.dynamodb_table(config.bm25_table_name())
    removed = _delete_indexed_document(tenant, doc_id, table)

    summary = {
        "key": key,
        "status": "not_indexed" if removed is None else "deleted",
        "tenant": tenant,
        "doc_id": doc_id,
        "chunks_removed": removed or 0,
    }
    log_json(logger, "document_deleted", **summary)
    return summary


# Retry safety. S3 invokes this function asynchronously, and Lambda retries a
# failed asynchronous invocation (two more attempts by default), so unexpected
# errors are logged and re-raised. Outcomes a retry can't change return
# normally: unsupported format, invalid tenant folder, no extractable text,
# and an object that no longer exists.
# Retries converge because every write is keyed by the document: re-ingestion
# first removes the previous version through its manifest, PutVectors and the
# BM25 postings overwrite the same keys, and a delete is a no-op once the
# manifest is gone.
# Residual effects of a partial failure:
#   * Each tenant-stats update (an ADD) is followed by a separate manifest
#     write or delete. If an attempt fails between the two, the retry applies
#     the stats update again, so the tenant's doc_count and total_len over- or
#     under-count by that document. Only BM25 scoring is skewed (N in idf and
#     the average chunk length); vectors, postings, and manifests converge.
#   * If the object is overwritten between attempts by a version with fewer
#     chunks, the failed attempt's chunks beyond the new chunk count aren't in
#     the new manifest: their postings remain until the document is next
#     re-ingested or deleted (DocIndex finds them), and their vectors remain
#     in the index and can still be returned by dense search, because cleanup
#     deletes only the vector keys the manifest records (the existing
#     non-atomic re-ingest limitation).
# Events that exhaust their retries go to the IngestionFailures SQS queue
# (stack output IngestionFailureQueueUrl), the function's on-failure
# destination, instead of being dropped.
def lambda_handler(event, _context):
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        event_name = record.get("eventName", "")
        try:
            if event_name.startswith("ObjectRemoved"):
                results.append(remove_object(bucket, key))
            else:
                results.append(ingest_object(bucket, key))
        except Exception as exc:
            log_json(logger, "ingestion_failed", level=40, key=key, event_name=event_name, error=str(exc))
            raise  # let Lambda's asynchronous retries apply
    return {"results": results}
