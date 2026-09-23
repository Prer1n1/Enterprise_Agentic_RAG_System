"""Pilot integration: Laya (github.com/NandhaKishorM/laya, PyPI `laya`) for
document category classification at ingestion — replaces an earlier Jev
(TypeSafe AI) pilot of the same shape. Also exposes laya_rerank() (below),
used ONLY by evaluation/ragas_eval.py to compare Laya's own relevance
scoring against Cohere's reranker as a THIRD evaluation column — not
wired into the live query path (retrieval/hybrid_retriever.py is
untouched), since the user explicitly asked to compare, not swap.

Why the swap: Jev's hosted API opened access on 2026-09-15 but was
waitlisted, then hit onboarding capacity limits the day access opened —
this project had no working key to verify against. Laya answers the exact
same typed Choice/Score/Noul question shape (same primitive names, same
{"type": "choice", "instructions": ..., "criteria": {...}} request shape
Jev uses) but runs as a local, open-weight model (Apache 2.0) — no API
key, no waitlist, no per-call cost, no external network dependency at
inference time. Verified as a real, legitimate project before adopting
it (not just trusting its README): checked its GitHub repo metadata via
the API directly (real commit history, real issues/PRs, Apache 2.0
license), confirmed `laya` is a real PyPI package with 16 published
releases under a verifiable author/company, then actually installed it,
downloaded its default checkpoint, and ran it — not just read about it.

Same decision-point reasoning as the Jev pilot it replaced: this fits
classify_category() specifically (pick one of a fixed category list) and
nothing else in this pipeline — it does not generate text, so synthesis
was never a candidate.

Real findings from live testing (see evaluation/laya_classification_eval.py
and docs/design-decisions.md for the full numbers):
  - 100% (6/6) accuracy on this project's hand-labeled excerpt set,
    beating the existing LLM tier's 83% (5/6).
  - The claimed ~33ms latency did NOT hold up on ordinary CPU hardware —
    real measured per-call inference here was ~900-1500ms, roughly on
    par with the existing OpenAI call, not a clear speed win on this
    hardware. Almost certainly a GPU-measured benchmark number.
  - The checkpoint self-reports a calibration problem on load
    ("temperatures outside [0.5, 5] would distort confidence... treat
    confidence from the affected buckets as uncalibrated") — and this
    showed up concretely: a correct answer scored confidence 0.372,
    which is why LAYA_CONFIDENCE_THRESHOLD below is 0.3, not the 0.6 the
    abandoned Jev pilot used. A higher threshold would silently discard
    correct answers from this specific checkpoint.
  - Still only days old (repo created 2026-09-18) — same "no long track
    record yet" caveat as Jev, just the open-weights flavor instead of
    the hosted-API flavor.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LAYA_MODEL_ID = "convaiinnovations/laya"

# Empirically chosen from a real test run, not copied from Jev's pilot:
# see this module's docstring for the 0.372-confidence correct answer
# that a 0.6 threshold would have wrongly discarded, and the checkpoint's
# own self-reported calibration warning.
LAYA_CONFIDENCE_THRESHOLD = 0.3

_agent = None
_agent_lock = threading.Lock()
_load_failed = False


def _get_agent():
    """Lazy singleton — loading the model takes real, measured time (about
    24s even from a warm on-disk cache) — must happen once per process,
    not once per document, or ingesting more than a handful of files
    would be dominated entirely by repeated model loads."""
    global _agent, _load_failed
    if _agent is not None or _load_failed:
        return _agent
    with _agent_lock:
        if _agent is not None or _load_failed:
            return _agent
        try:
            import laya

            _agent = laya.load(LAYA_MODEL_ID)
        except Exception:
            logger.exception("laya_load_failed")
            _load_failed = True
    return _agent


_RERANK_LEVELS = ["not relevant", "somewhat relevant", "highly relevant"]
_RERANK_MAX_LEVEL = len(_RERANK_LEVELS) - 1


def laya_rerank(query: str, documents: List[str], top_n: int) -> Optional[List[Tuple[int, float]]]:
    """Returns (index_into_documents, relevance_score) pairs, best-first,
    truncated to top_n, matching retrieval/reranker.py's rerank() contract
    exactly so it can be dropped into the same evaluation/comparison code.

    Scores a query/document pair with a `score` typed question (an ordinal
    scale, not Cohere's native continuous score) and normalizes Laya's own
    `score` field — a probability-weighted expected value over the ordinal
    levels, e.g. 0*P(not relevant) + 1*P(somewhat) + 2*P(highly), verified
    directly against a real response rather than assumed — into a 0-1
    range by dividing by the top level index, so it's on the same scale as
    Cohere's relevance_score for side-by-side comparison.

    THIS IS FOR THE EVALUATION HARNESS ONLY (see
    evaluation/ragas_eval.py's Laya reranking-impact column) — not wired
    into retrieval/hybrid_retriever.py's live query path. Returns None on
    ANY failure (no documents, model unavailable, malformed output),
    same fail-safe contract as the rest of this module. Never raises."""
    if not documents:
        return None
    agent = _get_agent()
    if agent is None:
        return None
    try:
        questions = {
            "relevance": {
                "type": "score",
                "instructions": f"How relevant is this document to answering the query: {query!r}?",
                "criteria": _RERANK_LEVELS,
            }
        }
        scored = []
        for i, doc in enumerate(documents):
            state = f"Query: {query}\n\nDocument: {doc}"
            result = agent.predict(state, questions)
            answer = result["answers"]["relevance"]
            normalized = float(answer["score"]) / _RERANK_MAX_LEVEL
            scored.append((i, normalized))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]
    except Exception:
        logger.exception("laya_rerank_failed")
        return None


def laya_classify_category(
    text: str,
    categories: List[str],
    criteria: Dict[str, str],
) -> Optional[Tuple[str, float]]:
    """Returns (category, confidence) or None on ANY failure — laya not
    installed, model failed to load, malformed output, or a category
    outside the known list. Same fallback contract as the Jev integration
    this replaced: the caller treats None exactly like a failed LLM call
    and falls through to the next tier. Never raises."""
    if not text.strip():
        return None
    agent = _get_agent()
    if agent is None:
        return None
    try:
        questions = {
            "category": {
                "type": "choice",
                "instructions": "Which single category best fits this document excerpt?",
                "criteria": criteria,
            }
        }
        result = agent.predict(text, questions)
        answer = result["answers"]["category"]
        category = answer["choice"]
        confidence = float(answer["confidence"])
        if category not in categories:
            return None
        if confidence < LAYA_CONFIDENCE_THRESHOLD:
            return None
        return category, confidence
    except Exception:
        logger.exception("laya_classify_failed")
        return None
