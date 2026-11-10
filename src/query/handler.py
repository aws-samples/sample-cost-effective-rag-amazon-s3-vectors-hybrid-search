"""Query Lambda: POST /query -> hybrid retrieval + grounded generation.

Flow: hybrid search (in-process) -> prompt with token budget -> Bedrock
Converse -> JSON answer with citations and token usage.

Bedrock practices applied (verified against current guidance):
  * Converse API (unified schema) rather than provider-native InvokeModel.
  * ``maxTokens`` ALWAYS set explicitly - unset values reserve the model
    maximum against the account quota and cause surprise throttling.
  * Cross-Region inference profile (default us.anthropic.claude-sonnet-4-6).
  * Adaptive retry mode on the shared client.

Streaming: REST APIs support the STREAM response transfer mode (proxy
integrations) since 2025-11-19, but the Python managed runtime can't stream a
response, so /query returns buffered JSON. ``stream_answer`` below is the
ConverseStream generator for a streaming front: a Node.js or Lambda Web
Adapter function behind a Function URL with RESPONSE_STREAM, or a REST method
in STREAM mode (see docs/architecture.md).
"""
import json
from typing import Iterator

from common import aws_clients, config
from common.logging_util import get_logger, log_json
from common.tenant import INVALID_TENANT_MESSAGE, resolve_request_tenant
from query.prompt_builder import build_messages
from search.hybrid import hybrid_search

logger = get_logger(__name__)

_HEADERS = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}
_MAX_OUTPUT_TOKENS_CAP = 4096


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": _HEADERS, "body": json.dumps(body, default=str)}


def _citations(contexts: list[dict]) -> list[dict]:
    return [
        {
            "index": i,
            "key": ctx.get("key"),
            "doc_id": (ctx.get("metadata") or {}).get("doc_id"),
            "source": (ctx.get("metadata") or {}).get("source"),
            "score": ctx.get("score"),
            "dense_rank": ctx.get("dense_rank"),
            "sparse_rank": ctx.get("sparse_rank"),
        }
        for i, ctx in enumerate(contexts, start=1)
    ]


def answer_question(
    question: str,
    tenant: str,
    top_k: int = 5,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> dict:
    """Full RAG round trip; returns the response payload as a dict."""
    results = hybrid_search(question, tenant, top_k=top_k)
    if not results:
        return {
            "answer": "I could not find any relevant documents for this question.",
            "citations": [],
            "model": config.generation_model_id(),
            "usage": {},
        }

    system_prompt, messages, included = build_messages(
        question, results, config.max_context_tokens()
    )

    response = aws_clients.bedrock_runtime().converse(
        modelId=config.generation_model_id(),
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
        },
    )

    content = response.get("output", {}).get("message", {}).get("content", [])
    answer = "".join(block.get("text", "") for block in content).strip()
    usage = response.get("usage", {})

    log_json(
        logger,
        "query_answered",
        tenant=tenant,
        contexts_used=len(included),
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
    )
    return {
        "answer": answer,
        "citations": _citations(included),
        "model": config.generation_model_id(),
        "usage": usage,
    }


def stream_answer(
    question: str,
    tenant: str,
    top_k: int = 5,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterator[str]:
    """Yield answer text deltas via ConverseStream.

    Not reachable through /query: the Python managed runtime can't stream a
    response, even though REST APIs support the STREAM response transfer
    mode (proxy integrations) since 2025-11-19. Token streaming needs a
    Node.js or Lambda Web Adapter front (Function URL with RESPONSE_STREAM,
    or a REST method in STREAM mode) - see docs/architecture.md.
    """
    results = hybrid_search(question, tenant, top_k=top_k)
    if not results:
        yield "I could not find any relevant documents for this question."
        return

    system_prompt, messages, _ = build_messages(question, results, config.max_context_tokens())
    response = aws_clients.bedrock_runtime().converse_stream(
        modelId=config.generation_model_id(),
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={"maxTokens": int(max_tokens), "temperature": float(temperature)},
    )
    for event in response["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            yield delta["text"]


def lambda_handler(event, _context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "request body must be valid JSON"})

    question = (body.get("question") or "").strip()
    if not question:
        return _response(400, {"error": "'question' is required"})

    # Blank or omitted -> default tenant; never an unscoped search.
    try:
        tenant = resolve_request_tenant(body.get("tenant_id"))
    except ValueError:
        return _response(400, {"error": INVALID_TENANT_MESSAGE})
    top_k = min(max(int(body.get("top_k") or 5), 1), 20)
    max_tokens = min(max(int(body.get("max_tokens") or 1024), 1), _MAX_OUTPUT_TOKENS_CAP)
    # NOTE: don't use `or` for temperature - an explicit 0.0 is valid and
    # would be silently replaced by the default.
    raw_temperature = body.get("temperature")
    temperature = 0.2 if raw_temperature is None else min(max(float(raw_temperature), 0.0), 1.0)

    try:
        payload = answer_question(
            question, tenant, top_k=top_k, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as exc:  # noqa: BLE001
        log_json(logger, "query_failed", level=40, error=str(exc))
        return _response(500, {"error": "internal error while answering"})

    return _response(200, payload)
