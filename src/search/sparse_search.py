"""Sparse retrieval path: BM25 (Okapi) over the DynamoDB inverted index.

Scoring (standard Okapi BM25, k1=1.2, b=0.75):
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(D, Q) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))

Data access per query: 1 GetItem (tenant stats) + 1 Query per unique term.
``df`` falls out of the postings read - no separate counter to drift.
"""
import math
from collections import defaultdict

from boto3.dynamodb.conditions import Key

from common import aws_clients, config
from common.text_utils import normalize_terms

K1 = 1.2
B = 0.75

# Bound the postings read per term: extremely common terms carry little BM25
# signal (low idf) but can be arbitrarily expensive to read exhaustively.
MAX_POSTINGS_PER_TERM = 5000
_PAGE_LIMIT = 1000


def _stats(table, tenant: str) -> tuple[int, float]:
    item = table.get_item(Key={"PK": f"T#{tenant}#STATS", "SK": "GLOBAL"}).get("Item") or {}
    n = int(item.get("doc_count", 0))
    total = float(item.get("total_len", 0))
    return n, (total / n if n else 0.0)


def _term_postings(table, tenant: str, term: str) -> list[dict]:
    postings: list[dict] = []
    kwargs = {
        "KeyConditionExpression": Key("PK").eq(f"T#{tenant}#TERM#{term}"),
        "Limit": _PAGE_LIMIT,
    }
    while True:
        page = table.query(**kwargs)
        postings.extend(page.get("Items", []))
        if "LastEvaluatedKey" not in page or len(postings) >= MAX_POSTINGS_PER_TERM:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return postings[:MAX_POSTINGS_PER_TERM]


def bm25_search(query: str, tenant: str, top_k: int = 10, *, table=None) -> list[dict]:
    """Return BM25-ranked results: [{key, score}] (best first)."""
    table = table if table is not None else aws_clients.dynamodb_table(config.bm25_table_name())

    terms = sorted(set(normalize_terms(query)))
    if not terms:
        return []

    n_docs, avgdl = _stats(table, tenant)
    if n_docs == 0 or avgdl == 0:
        return []

    scores: dict[str, float] = defaultdict(float)
    for term in terms:
        postings = _term_postings(table, tenant, term)
        df = len(postings)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for posting in postings:
            chunk_key = str(posting["SK"])[len("CHUNK#"):]
            tf = float(posting["tf"])
            dl = float(posting["dl"])
            denom = tf + K1 * (1.0 - B + B * dl / avgdl)
            scores[chunk_key] += idf * tf * (K1 + 1.0) / denom

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"key": key, "score": score} for key, score in ranked[:top_k]]
