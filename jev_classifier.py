"""Pilot integration: TypeSafe AI's Jev for document category
classification at ingestion.

Why this decision point and no other: Jev ("System One") isn't a
conversational LLM — it takes typed questions (Choice/Score/Noul) against
a piece of program state and returns a calibrated, schema-constrained
answer in a single non-autoregressive pass, not free text. That shape
matches exactly one decision already in this pipeline: classify_category()
picking ONE category from a FIXED list (agent/planner.py's routing
decision is the other close fit, but that one also produces the `reason`
free-text field the router doesn't need here, and is on the hot query
path where an unproven week-old provider is a worse place to experiment
first than the offline ingestion path is). It is NOT a fit for synthesis
(agent/nodes.py) — Jev does not generate text — so it was never
considered for that step.

Opt-in only (use_jev=False by default everywhere this wires in — see
ingestion/metadata_extractor.py and ingestion/pipeline.py): this pilot
adds a NEW top tier ahead of the existing, already-verified LLM
classifier, not a replacement for it. Any failure (no TYPESAFE_API_KEY,
network error, malformed response, exhausted retries) or low-confidence
answer falls through to the existing LLM -> keyword chain untouched —
same graceful-degradation shape as retrieval/reranker.py's Cohere
integration.

No official SDK dependency added for a single-endpoint pilot: one
POST /v1/systemone call over httpx (already a transitive dependency via
openai/cohere) is simpler to reason about and verify than trusting an
unfamiliar SDK's exact method signatures for a still-early-access API.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import httpx

from config import JEV_MODEL, TYPESAFE_API_KEY
from retry_utils import retry_jev_call

logger = logging.getLogger(__name__)

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

# Jev returns a probability per option alongside its top choice. A choice
# made with low confidence is worse than no Jev answer at all for a step
# that already has a proven fallback sitting right behind it — below this,
# treat it the same as a failure and let the LLM tier decide instead.
JEV_CONFIDENCE_THRESHOLD = 0.6

_TIMEOUT_SECONDS = 10.0
_QUESTION_KEY = "category"


@retry_jev_call
def _call_jev(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
        response = client.post(JEV_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def jev_classify_category(
    text: str,
    categories: List[str],
    criteria: Dict[str, str],
    max_chars: int = 3000,
) -> Optional[Tuple[str, float]]:
    """Returns (category, confidence) or None on ANY failure — no
    TYPESAFE_API_KEY configured, network error, exhausted retries, a
    response shape that doesn't parse, or a returned category outside the
    known list. The caller (metadata_extractor.py) treats None exactly
    like a failed LLM call: fall through to the next tier. Never raises."""
    if not TYPESAFE_API_KEY or not text.strip():
        return None
    try:
        payload = {
            "model": JEV_MODEL,
            "state": text[:max_chars],
            "questions": {
                _QUESTION_KEY: {
                    "type": "choice",
                    "instructions": "Which single category best fits this document excerpt?",
                    "criteria": criteria,
                }
            },
        }
        body = _call_jev(payload)
        answer = body["answers"][_QUESTION_KEY]
        category = answer["choice"]
        confidence = float(answer["confidence"])
        if category not in categories:
            return None
        if confidence < JEV_CONFIDENCE_THRESHOLD:
            return None
        return category, confidence
    except Exception:
        logger.exception("jev_classify_failed")
        return None
