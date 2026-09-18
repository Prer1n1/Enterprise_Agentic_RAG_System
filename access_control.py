"""Role-based access control — extends the single-shared-key auth built in
api/security.py with per-key SCOPES: which knowledge-source categories a
key can query, and whether it can ingest documents at all.

The existing API_KEY (config.py) is always the built-in "admin" scope —
full access, can ingest, queries every category — so nothing about the
original Authentication feature breaks. Additional, more restricted keys
are defined in an optional JSON file (ACCESS_CONTROL_CONFIG_PATH). No file
configured -> no additional keys, just the one admin key. Same
"config-only, degrades to the simple baseline" pattern as LangSmith/Drive/
Cohere elsewhere in this project.

Example config file (see docs/design-decisions.md for the full story):

    {
      "keys": [
        {"key": "<a generated secret>", "role": "hr-team",
         "categories": ["HR", "General"], "can_ingest": false}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import ACCESS_CONTROL_CONFIG_PATH, API_KEY

logger = logging.getLogger(__name__)

ADMIN_ROLE = "admin"
ALL_CATEGORIES = "*"


@dataclass(frozen=True)
class KeyScope:
    role: str
    categories: List[str]  # ["*"] means unrestricted
    can_ingest: bool

    def filter_categories(self, categories: List[str]) -> List[str]:
        if ALL_CATEGORIES in self.categories:
            return list(categories)
        return [c for c in categories if c in self.categories]


_ADMIN_SCOPE = KeyScope(role=ADMIN_ROLE, categories=[ALL_CATEGORIES], can_ingest=True)


def _load_scoped_keys() -> Dict[str, KeyScope]:
    """Loads ADDITIONAL scoped keys from ACCESS_CONTROL_CONFIG_PATH, on top
    of the always-present admin key. A missing/unset file is not an error —
    it just means no additional keys exist yet."""
    if not ACCESS_CONTROL_CONFIG_PATH:
        return {}
    try:
        with open(ACCESS_CONTROL_CONFIG_PATH) as f:
            raw = json.load(f)
        scopes = {}
        for entry in raw.get("keys", []):
            scopes[entry["key"]] = KeyScope(
                role=entry["role"],
                categories=entry.get("categories", []),
                can_ingest=entry.get("can_ingest", False),
            )
        return scopes
    except Exception:
        logger.exception("access_control_config_load_failed")
        return {}


# Loaded once at import time — same as the ingestion tracker's DB path or
# the Drive connector's credentials: cheap to load, no reason to re-read
# the file on every request.
_SCOPED_KEYS: Dict[str, KeyScope] = _load_scoped_keys()


def resolve_scope(provided_key: str) -> Optional[KeyScope]:
    """Returns the KeyScope for a valid key, or None if it doesn't match
    anything. The admin key — the highest-value credential — is checked
    with secrets.compare_digest() for a constant-time comparison, same as
    the original single-key implementation. Scoped keys use a direct dict
    lookup: an acceptable, standard tradeoff for a larger, lower-privilege,
    individually-revocable set of keys (a real system with many scoped
    keys would typically hash-index them anyway, which is what a dict
    lookup effectively does)."""
    if API_KEY and secrets.compare_digest(provided_key, API_KEY):
        return _ADMIN_SCOPE
    return _SCOPED_KEYS.get(provided_key)
