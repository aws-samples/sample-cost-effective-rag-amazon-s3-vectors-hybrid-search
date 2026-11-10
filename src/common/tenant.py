"""Tenant ID rule shared by ingestion and the /search and /query handlers.

A tenant ID is 1-64 characters of lowercase letters, digits, hyphens, and
underscores, starting with a letter or digit: ``^[a-z0-9][a-z0-9_-]{0,63}$``.

* Ingestion: the folder name in ``tenants/<tenant>/...`` is lowercased and
  must match the rule; uploads under a folder that doesn't are rejected.
* /search and /query: ``tenant_id`` is trimmed and lowercased. Omitted,
  empty, or whitespace-only means the default tenant (never "no filter").
  Anything else that doesn't match the rule is rejected with HTTP 400.

The request JSON Schema in ``infra/stacks/api_stack.py``
(``TENANT_ID_SCHEMA_PATTERN``) accepts exactly the values that
``resolve_request_tenant`` accepts; keep the two consistent.
"""
import re

from common import config

TENANT_ID_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"
_TENANT_ID_RE = re.compile(TENANT_ID_PATTERN)

INVALID_TENANT_MESSAGE = (
    "'tenant_id' must match ^[a-z0-9][a-z0-9_-]{0,63}$ "
    "(surrounding whitespace is trimmed and letters are lowercased)"
)


def is_valid_tenant_id(value) -> bool:
    """True if ``value`` is already a normalized, valid tenant ID."""
    # fullmatch (not match + "$"), so a trailing newline can't slip through.
    return isinstance(value, str) and _TENANT_ID_RE.fullmatch(value) is not None


def resolve_request_tenant(raw) -> str:
    """Normalize a request's ``tenant_id``; raise ValueError if it's invalid."""
    if raw is None:
        return config.default_tenant()
    if not isinstance(raw, str):
        raise ValueError(INVALID_TENANT_MESSAGE)
    tenant = raw.strip().lower()
    if not tenant:
        return config.default_tenant()
    if not is_valid_tenant_id(tenant):
        raise ValueError(INVALID_TENANT_MESSAGE)
    return tenant
