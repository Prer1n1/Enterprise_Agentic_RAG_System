"""Metadata extractor — enriches what loaders already capture (source,
title, author, section/page) with information no loader can infer purely
from file structure: a stable document ID, word count, detected language,
and a content category.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from langdetect import LangDetectException, detect

from .schema import Document

# Rule-based, not ML: cheap, deterministic, and easy to explain/debug.
# Production alternative: zero-shot LLM classification or a small fine-tuned
# classifier for documents that mix topics or use domain-specific wording
# these keyword lists don't cover.
_CATEGORY_KEYWORDS = {
    "HR": {"leave", "employee", "onboarding", "payroll", "benefits", "hiring", "remote work"},
    "Finance": {"expense", "budget", "invoice", "payment", "reimbursement", "cost"},
    "Security": {"password", "authentication", "incident", "access control", "vpn", "breach", "mfa"},
    "IT": {"system", "server", "software", "network", "deployment", "infrastructure"},
    "Legal": {"contract", "liability", "compliance", "agreement", "terms"},
}

# Public: the agent's router (agent/planner.py) reuses this list, so the
# set of "knowledge sources" it can route to can never drift out of sync
# with the categories documents actually get tagged with.
CATEGORIES = list(_CATEGORY_KEYWORDS.keys()) + ["General"]

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# langdetect is unreliable below ~20 words (verified: a 12-word English CSV
# row was misclassified as French in testing). Below this threshold we
# don't trust it — default to the platform's assumed corpus language
# instead of reporting false confidence.
_MIN_WORDS_FOR_LANG_DETECT = 20
_DEFAULT_LANGUAGE = "en"


def stable_doc_id(source: str) -> str:
    """Stable ID for the ORIGINAL source file (not per-chunk/section) —
    every page/section/row loaded from the same file shares this ID, so
    they can be traced back to one document for citation grouping or dedup."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def detect_language(text: str, word_count: int) -> str:
    if word_count < _MIN_WORDS_FOR_LANG_DETECT:
        return _DEFAULT_LANGUAGE
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"  # no alphabetic content to detect from


def classify_category(text: str, hint: Optional[str] = None) -> str:
    """hint: an explicit category from structured source data (e.g. a
    CSV's "department" column). Trusted outright when it's a known
    category — ground-truth structured data should never be overridden
    by keyword-matching free text (verified bug: a CSV row with
    department=IT about "password rotation" was misclassified as
    Security by keywords alone, since "password" is a Security keyword)."""
    if hint and hint in CATEGORIES:
        return hint

    lowered = text.lower()
    scores = {
        category: sum(1 for kw in keywords if kw in lowered)
        for category, keywords in _CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_category if best_score > 0 else "General"


def enrich(document: Document) -> Document:
    """Adds doc_id, word_count, language, and category into
    metadata.extra. Mutates and returns the same Document."""
    extra = document.metadata.extra
    extra.setdefault("doc_id", stable_doc_id(document.metadata.source))
    word_count = len(_WORD_RE.findall(document.content))
    extra["word_count"] = word_count
    extra["language"] = detect_language(document.content, word_count)
    extra["category"] = classify_category(document.content, hint=extra.get("category_hint"))
    return document


def enrich_all(documents: List[Document]) -> List[Document]:
    return [enrich(doc) for doc in documents]
