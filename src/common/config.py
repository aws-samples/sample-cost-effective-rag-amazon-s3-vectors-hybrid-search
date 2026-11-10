"""Runtime configuration for the Lambda functions (12-factor: env vars only)."""
import os


def vector_bucket_name() -> str:
    return os.environ["VECTOR_BUCKET_NAME"]


def vector_index_name() -> str:
    return os.environ["VECTOR_INDEX_NAME"]


def bm25_table_name() -> str:
    return os.environ["BM25_TABLE_NAME"]


def embedding_model_id() -> str:
    return os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")


def embedding_dimensions() -> int:
    return int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))


def generation_model_id() -> str:
    return os.environ.get("GENERATION_MODEL_ID", "us.anthropic.claude-sonnet-4-6")


def default_tenant() -> str:
    return os.environ.get("DEFAULT_TENANT", "default")


def max_context_tokens() -> int:
    """Token budget for retrieved context inside the generation prompt."""
    return int(os.environ.get("MAX_CONTEXT_TOKENS", "6000"))


def chunk_chars() -> int:
    """Target chunk size in characters (~4 chars/token; 2048 ~= 512 tokens)."""
    return int(os.environ.get("CHUNK_CHARS", "2048"))


def chunk_overlap_chars() -> int:
    """Overlap seeded into the next chunk (~12.5% of the default chunk size)."""
    return int(os.environ.get("CHUNK_OVERLAP_CHARS", "256"))
