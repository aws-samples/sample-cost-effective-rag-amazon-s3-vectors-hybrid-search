"""BM25 inverted index maintenance in DynamoDB.

Item model (single table):
    Posting : PK=T#<tenant>#TERM#<term>  SK=CHUNK#<chunk_key>
              attrs: tf (term frequency), dl (chunk length in tokens),
                     gsi1pk=T#<tenant>#DOC#<doc_id>   [GSI: DocIndex, KEYS_ONLY]
    Stats   : PK=T#<tenant>#STATS        SK=GLOBAL
              attrs: doc_count (chunks), total_len (sum of dl)
    Manifest: PK=T#<tenant>#DOC#<doc_id> SK=MANIFEST
              attrs: chunk_count, total_len, s3_key, ingested_at

Document frequency (df) is derived at query time as the number of postings
returned for a term - no separate counter to keep consistent.
"""
from collections import Counter

from boto3.dynamodb.conditions import Key

from common import aws_clients, config
from common.text_utils import normalize_terms


def stats_key(tenant: str) -> dict:
    return {"PK": f"T#{tenant}#STATS", "SK": "GLOBAL"}


def manifest_key(tenant: str, doc_id: str) -> dict:
    return {"PK": f"T#{tenant}#DOC#{doc_id}", "SK": "MANIFEST"}


def compute_postings(text: str) -> tuple[Counter, int]:
    """Pure function: term frequencies and token length for one chunk."""
    terms = normalize_terms(text)
    return Counter(terms), len(terms)


def index_chunks(tenant: str, doc_id: str, chunk_keys: list[str], texts: list[str], *, table=None) -> int:
    """Write postings for all chunks of a document and bump tenant stats.

    Returns the total token length indexed (stored on the manifest so a
    re-ingest can decrement stats precisely).
    """
    table = table if table is not None else aws_clients.dynamodb_table(config.bm25_table_name())
    doc_partition = f"T#{tenant}#DOC#{doc_id}"

    total_len = 0
    with table.batch_writer() as batch:
        for chunk_key, text in zip(chunk_keys, texts):
            tf_map, dl = compute_postings(text)
            total_len += dl
            for term, tf in tf_map.items():
                batch.put_item(
                    Item={
                        "PK": f"T#{tenant}#TERM#{term}",
                        "SK": f"CHUNK#{chunk_key}",
                        "tf": int(tf),
                        "dl": int(dl),
                        "gsi1pk": doc_partition,
                    }
                )

    table.update_item(
        Key=stats_key(tenant),
        UpdateExpression="ADD doc_count :dc, total_len :tl",
        ExpressionAttributeValues={":dc": len(chunk_keys), ":tl": int(total_len)},
    )
    return total_len


def delete_document_postings(tenant: str, doc_id: str, *, table=None) -> int:
    """Remove all postings for a document (found via the DocIndex GSI).

    Stats are decremented by the caller using the manifest's recorded
    chunk_count/total_len (this function only deletes postings).
    Returns the number of posting items deleted.
    """
    table = table if table is not None else aws_clients.dynamodb_table(config.bm25_table_name())
    doc_partition = f"T#{tenant}#DOC#{doc_id}"

    deleted = 0
    kwargs = {
        "IndexName": "DocIndex",
        "KeyConditionExpression": Key("gsi1pk").eq(doc_partition),
    }
    while True:
        page = table.query(**kwargs)
        items = page.get("Items", [])
        if items:
            with table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})
            deleted += len(items)
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return deleted


def decrement_stats(tenant: str, chunk_count: int, total_len: int, *, table=None) -> None:
    table = table if table is not None else aws_clients.dynamodb_table(config.bm25_table_name())
    table.update_item(
        Key=stats_key(tenant),
        UpdateExpression="ADD doc_count :dc, total_len :tl",
        ExpressionAttributeValues={":dc": -int(chunk_count), ":tl": -int(total_len)},
    )
