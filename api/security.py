"""API-key authentication for the FastAPI layer.

Uses FastAPI's own Security dependency system (`APIKeyHeader`) rather than
hand-parsing a header — the whole point of "don't hand-roll it" applies to
auth infrastructure just as much as to metrics/evaluation.

require_api_key returns the matched key's KeyScope (access_control.py),
not just True/False — route handlers declare it as a parameter
(`scope: KeyScope = Depends(require_api_key)`) to enforce per-key
permissions (can this key ingest? which categories can it query?), not
just "is this key valid at all."
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from access_control import KeyScope, resolve_scope

logger = logging.getLogger(__name__)

# auto_error=False: a missing header should fall through to OUR error
# message below (mentioning the exact header name), not FastAPI's generic
# 403 with no explanation.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, provided: Optional[str] = Security(_api_key_header)) -> KeyScope:
    scope = resolve_scope(provided) if provided else None
    if scope is None:
        # Log the failed attempt for security monitoring — but never the
        # provided value itself, wrong or not. Logging "almost right"
        # guesses is its own minor leak, and logging the correct key on a
        # rare false negative would be worse.
        logger.warning(
            "auth_failed",
            extra={"path": request.url.path, "client": request.client.host if request.client else None},
        )
        raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header)")
    return scope
