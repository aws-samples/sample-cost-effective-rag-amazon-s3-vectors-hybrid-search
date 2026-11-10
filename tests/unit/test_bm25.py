from ingestion.bm25_indexer import compute_postings, decrement_stats, delete_document_postings, index_chunks
from search.sparse_search import bm25_search


def _index_corpus(table):
    docs = {
        "doc-a": [
            "DynamoDB is a serverless key value database. DynamoDB scales automatically.",
            "Vector search finds semantically similar passages using embeddings.",
        ],
        "doc-b": [
            "S3 Vectors stores vector embeddings cheaply for retrieval augmented generation.",
        ],
        "doc-c": [
            "Completely unrelated text about cooking pasta with tomatoes and basil.",
        ],
    }
    for doc_id, chunks in docs.items():
        keys = [f"default#{doc_id}#{i:05d}" for i in range(len(chunks))]
        index_chunks("default", doc_id, keys, chunks, table=table)


def test_compute_postings_counts_and_length():
    tf, dl = compute_postings("DynamoDB scales. DynamoDB is serverless and fast.")
    assert tf["dynamodb"] == 2
    assert dl == sum(tf.values())
    assert "is" not in tf and "and" not in tf  # stopwords excluded


def test_stats_accumulate_across_documents(fake_table):
    _index_corpus(fake_table)
    stats = fake_table.get_item(Key={"PK": "T#default#STATS", "SK": "GLOBAL"})["Item"]
    assert stats["doc_count"] == 4  # 4 chunks total
    assert stats["total_len"] > 0


def test_bm25_ranks_relevant_chunk_first(fake_table):
    _index_corpus(fake_table)
    results = bm25_search("dynamodb serverless database", "default", top_k=3, table=fake_table)
    assert results, "expected at least one match"
    assert results[0]["key"] == "default#doc-a#00000"
    keys = [r["key"] for r in results]
    assert "default#doc-c#00000" not in keys  # pasta chunk shares no terms


def test_bm25_query_with_no_matching_terms(fake_table):
    _index_corpus(fake_table)
    assert bm25_search("quantum chromodynamics", "default", table=fake_table) == []


def test_bm25_empty_tenant_returns_empty(fake_table):
    _index_corpus(fake_table)
    assert bm25_search("dynamodb", "other-tenant", table=fake_table) == []


def test_tenant_key_prefix_isolates_postings(fake_table):
    index_chunks("anyexamplecompany", "d1", ["anyexamplecompany#d1#00000"], ["shared secret roadmap"], table=fake_table)
    index_chunks("example-corp", "d1", ["example-corp#d1#00000"], ["shared secret roadmap"], table=fake_table)
    anyexamplecompany = bm25_search("secret roadmap", "anyexamplecompany", table=fake_table)
    assert [r["key"] for r in anyexamplecompany] == ["anyexamplecompany#d1#00000"]


def test_delete_document_postings_and_stats(fake_table):
    _index_corpus(fake_table)
    deleted = delete_document_postings("default", "doc-a", table=fake_table)
    assert deleted > 0
    decrement_stats("default", 2, 20, table=fake_table)

    results = bm25_search("dynamodb serverless", "default", table=fake_table)
    assert all(not r["key"].startswith("default#doc-a#") for r in results)


# Identifiers that differ only in their last segment. The near-miss also
# mentions ZX-BETA-9, so it contains "zx" twice as often as the right document
# and "9" just as often: single-digit terms alone would rank it first.
FIRMWARE_CURRENT = (
    "# Firmware release notes - current train\n## ZX-ALPHA-9\n"
    "Release ZX-ALPHA-9 adds secure boot verification and fixes the fan speed oscillation\n"
    "seen under sustained load. Upgrade path: install directly from any 8.x release."
)
FIRMWARE_PREVIOUS = (
    "# Firmware release notes - previous train\n## ZX-ALPHA-8\n"
    "Release ZX-ALPHA-8 introduced a regression where the management interface stopped\n"
    "responding after 30 days of uptime. Fixed in a later release.\n## ZX-BETA-9\n"
    "Beta build ZX-BETA-9 tested the new boot loader; not for production."
)


def test_bm25_ranks_each_identifier_own_document_first(fake_table):
    index_chunks("default", "fw-current", ["default#fw-current#00000"], [FIRMWARE_CURRENT], table=fake_table)
    index_chunks("default", "fw-previous", ["default#fw-previous#00000"], [FIRMWARE_PREVIOUS], table=fake_table)

    for query, expected in (
        ("ZX-ALPHA-9", "default#fw-current#00000"),
        ("ZX-ALPHA-8", "default#fw-previous#00000"),
    ):
        results = bm25_search(query, "default", table=fake_table)
        assert results[0]["key"] == expected, query
        assert results[0]["score"] > results[1]["score"], query  # a real win, not a key-order tie
