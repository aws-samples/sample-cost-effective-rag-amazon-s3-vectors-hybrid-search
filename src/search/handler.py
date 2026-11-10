"""Search Lambda: POST /search -> hybrid retrieval (no LLM call)."""
import json

from common.logging_util import get_logger, log_json
from common.tenant import INVALID_TENANT_MESSAGE, resolve_request_tenant
from search.hybrid import VALID_MODES, hybrid_search

logger = get_logger(__name__)

_HEADERS = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": _HEADERS, "body": json.dumps(body, default=str)}


def lambda_handler(event, _context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "request body must be valid JSON"})

    query = (body.get("query") or "").strip()
    if not query:
        return _response(400, {"error": "'query' is required"})

    # Blank or omitted -> default tenant; never an unscoped search.
    try:
        tenant = resolve_request_tenant(body.get("tenant_id"))
    except ValueError:
        return _response(400, {"error": INVALID_TENANT_MESSAGE})
    top_k = min(max(int(body.get("top_k") or 5), 1), 50)
    mode = body.get("mode") or "hybrid"
    if mode not in VALID_MODES:
        return _response(400, {"error": f"'mode' must be one of {list(VALID_MODES)}"})

    try:
        results = hybrid_search(query, tenant, top_k=top_k, mode=mode)
    except Exception as exc:  # noqa: BLE001
        log_json(logger, "search_failed", level=40, error=str(exc))
        return _response(500, {"error": "internal error while searching"})

    return _response(200, {"query": query, "tenant_id": tenant, "mode": mode, "results": results})
