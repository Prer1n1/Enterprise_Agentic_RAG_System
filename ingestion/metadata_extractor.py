"""Metadata extractor — enriches what loaders already capture (source,
title, author, section/page) with information no loader can infer purely
from file structure: a stable document ID, word count, detected language,
and a content category.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field

from laya_classifier import laya_classify_category
from retry_utils import retry_openai_call

from .schema import Document

# Keyword matching is cheap, deterministic, and easy to explain/debug — kept
# as the FALLBACK classifier (used when the LLM call fails, and directly
# testable with zero API cost), not as the primary path anymore. Verified
# limitation that motivated adding an LLM primary: a real SEC investor
# bulletin scored 0 Finance keyword hits across 26 chunks (17 Legal, 6
# General, 3 HR instead) because its real-world vocabulary ("offering",
# "prospectus", "securities") never overlaps with this list's
# synthetic-sample-data terms ("expense", "budget", "reimbursement"). A
# keyword list tuned to one corpus doesn't generalize — see
# docs/design-decisions.md.
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

# Laya's Choice primitive takes a short description per option, not a raw
# keyword set — reusing _CATEGORY_KEYWORDS (the same real definitions the
# keyword fallback and the LLM prompt both draw from, so all three
# classifiers stay in sync with one definition of what each category means)
# rather than writing a fourth, separately-drifting description of "HR".
_LAYA_CRITERIA = {cat: f"about {', '.join(sorted(kws))}" for cat, kws in _CATEGORY_KEYWORDS.items()}
_LAYA_CRITERIA["General"] = "none of the other categories clearly apply"

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# langdetect is unreliable below ~20 words (verified: a 12-word English CSV
# row was misclassified as French in testing). Below this threshold we
# don't trust it — default to the platform's assumed corpus language
# instead of reporting false confidence.
_MIN_WORDS_FOR_LANG_DETECT = 20
_DEFAULT_LANGUAGE = "en"

_CLASSIFIER_MODEL = "gpt-4o-mini"
# Category classification doesn't need a whole page/section of text to
# decide a topic — capping keeps cost down without hurting accuracy.
_MAX_CHARS_FOR_CLASSIFICATION = 3000


class CategoryDecision(BaseModel):
    category: str = Field(
        description=(
            f"The single best-fitting category for this document excerpt. "
            f"Must be exactly one of: {CATEGORIES}. Use 'General' only if "
            f"none of the specific categories genuinely fit."
        )
    )


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


def _keyword_classify_category(text: str) -> str:
    """The original rule-based classifier. Cheap, deterministic, zero API
    cost — kept as the fallback for when the LLM call fails (network error,
    missing API key, etc.), so a transient failure degrades to "less
    accurate" rather than "ingestion breaks"."""
    lowered = text.lower()
    scores = {
        category: sum(1 for kw in keywords if kw in lowered)
        for category, keywords in _CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_category if best_score > 0 else "General"


@retry_openai_call
def _invoke_classifier(structured_llm, prompt: str) -> "CategoryDecision":
    """Retried a few times before _llm_classify_category gives up and
    falls back to the keyword classifier below — a transient blip
    shouldn't cost classification quality when retrying would have
    succeeded. See retry_utils.py."""
    return structured_llm.invoke(prompt)


def _llm_classify_category(text: str, llm=None) -> Optional[str]:
    """Zero-shot LLM classification — the fix for the keyword classifier's
    real-world vocabulary gap (see _CATEGORY_KEYWORDS' docstring above).
    Returns None on ANY failure (bad API key, network error, malformed
    output) so the caller falls back to the keyword classifier instead of
    letting a transient LLM failure break ingestion outright."""
    try:
        llm = llm or ChatOpenAI(model=_CLASSIFIER_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(CategoryDecision)
        decision = _invoke_classifier(
            structured_llm,
            f"Categories: {CATEGORIES}\n\n"
            f"Document excerpt:\n{text[:_MAX_CHARS_FOR_CLASSIFICATION]}\n\n"
            f"Which single category best fits this content?"
        )
        return decision.category if decision.category in CATEGORIES else None
    except Exception:
        return None


def _laya_classify_category(text: str) -> Optional[str]:
    """Wraps laya_classifier.laya_classify_category with this module's own
    CATEGORIES/_LAYA_CRITERIA — kept here rather than in laya_classifier.py
    so that module stays generic (no dependency on this module's category
    taxonomy) and this file stays the one place CATEGORIES is defined.
    Returns None on any failure or low-confidence answer; see
    laya_classifier.py's own docstring for the full fallback reasoning."""
    result = laya_classify_category(text, CATEGORIES, _LAYA_CRITERIA)
    return result[0] if result else None


def classify_category(
    text: str,
    hint: Optional[str] = None,
    llm=None,
    use_llm: bool = True,
    use_laya: bool = False,
) -> str:
    """hint: an explicit category from structured source data (e.g. a
    CSV's "department" column). Trusted outright when it's a known
    category — ground-truth structured data should never be overridden
    by classification of free text (verified bug: a CSV row with
    department=IT about "password rotation" was misclassified as
    Security by keywords alone, since "password" is a Security keyword).

    use_laya=True tries the Laya pilot FIRST (see laya_classifier.py) —
    opt-in, defaults False, never replaces the LLM tier below it, only
    ever runs ahead of it. use_llm=False forces the free/offline keyword
    path — used by tests that need to stay fast, deterministic, and cost
    nothing."""
    if hint and hint in CATEGORIES:
        return hint

    if use_laya:
        laya_result = _laya_classify_category(text)
        if laya_result:
            return laya_result

    if use_llm:
        llm_result = _llm_classify_category(text, llm=llm)
        if llm_result:
            return llm_result

    return _keyword_classify_category(text)


def enrich(document: Document, llm=None, use_llm: bool = True, use_laya: bool = False) -> Document:
    """Adds doc_id, word_count, language, and category into
    metadata.extra. Mutates and returns the same Document."""
    extra = document.metadata.extra
    extra.setdefault("doc_id", stable_doc_id(document.metadata.source))
    word_count = len(_WORD_RE.findall(document.content))
    extra["word_count"] = word_count
    extra["language"] = detect_language(document.content, word_count)
    extra["category"] = classify_category(
        document.content, hint=extra.get("category_hint"), llm=llm, use_llm=use_llm, use_laya=use_laya
    )
    return document


def enrich_all(documents: List[Document], llm=None, use_llm: bool = True, use_laya: bool = False) -> List[Document]:
    return [enrich(doc, llm=llm, use_llm=use_llm, use_laya=use_laya) for doc in documents]
