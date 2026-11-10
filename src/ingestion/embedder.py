"""Amazon Titan Text Embeddings V2 wrapper.

Verified request/response contract (Bedrock InvokeModel, model
``amazon.titan-embed-text-v2:0``):
    request : {"inputText": str, "dimensions": 1024|512|256, "normalize": bool}
    response: {"embedding": [float, ...], "inputTextTokenCount": int}
Limits: 8,192 tokens / 50,000 characters of input; no batch API (loop).
"""
import json

from common import aws_clients, config

_TITAN_MAX_CHARS = 50000


def embed_text(text: str, dimensions: int | None = None, normalize: bool = True) -> list[float]:
    """Return the embedding vector for ``text``.

    Values are coerced to Python floats; S3 Vectors stores float32 and the
    SDK converts on write.
    """
    body = {
        "inputText": text[:_TITAN_MAX_CHARS],
        "dimensions": int(dimensions or config.embedding_dimensions()),
        "normalize": normalize,
    }
    response = aws_clients.bedrock_runtime().invoke_model(
        modelId=config.embedding_model_id(),
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read())
    return [float(v) for v in payload["embedding"]]


def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    """Embed each text sequentially (Titan exposes no batch endpoint)."""
    return [embed_text(t, dimensions=dimensions) for t in texts]
