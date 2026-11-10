"""Typed configuration for the pattern, sourced from CDK context (cdk.json / -c)."""
from dataclasses import dataclass

import aws_cdk as cdk


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PatternConfig:
    prefix: str
    enable_knowledge_base: bool
    embedding_model_id: str
    embedding_dimensions: int
    generation_model_id: str
    vector_index_name: str
    kb_vector_index_name: str
    api_throttle_rate: int
    api_throttle_burst: int
    api_monthly_quota: int

    @classmethod
    def from_context(cls, app: cdk.App) -> "PatternConfig":
        ctx = app.node.try_get_context
        return cls(
            prefix=str(ctx("prefix") or "hybrid-rag"),
            enable_knowledge_base=_as_bool(ctx("enable_knowledge_base") or False),
            embedding_model_id=str(ctx("embedding_model_id") or "amazon.titan-embed-text-v2:0"),
            # Titan Text Embeddings V2 supports 1024 (default), 512, or 256.
            embedding_dimensions=int(ctx("embedding_dimensions") or 1024),
            # Cross-Region inference profile. Cost-optimized alternative:
            #   us.anthropic.claude-haiku-4-5-20251001-v1:0 (one-third of Sonnet
            #   4.6's per-token price; Active on Amazon Bedrock with no announced
            #   end-of-life date and at least 6 months' Legacy notice before EOL;
            #   check the model card before switching).
            generation_model_id=str(ctx("generation_model_id") or "us.anthropic.claude-sonnet-4-6"),
            vector_index_name=str(ctx("vector_index_name") or "chunks"),
            kb_vector_index_name=str(ctx("kb_vector_index_name") or "kb-chunks"),
            api_throttle_rate=int(ctx("api_throttle_rate") or 10),
            api_throttle_burst=int(ctx("api_throttle_burst") or 20),
            api_monthly_quota=int(ctx("api_monthly_quota") or 50000),
        )
