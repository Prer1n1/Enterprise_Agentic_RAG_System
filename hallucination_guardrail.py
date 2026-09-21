"""Live hallucination guardrail — scores each synthesized answer's RAGAS
Faithfulness in real time and flags it if below HALLUCINATION_THRESHOLD.

This is the exact same metric the offline evaluation pipeline
(evaluation/ragas_eval.py) uses — same metric, same threshold — just
applied to ONE live answer instead of a batch eval set. A low faithfulness
score means at least one claim in the answer isn't supported by the
retrieved context, which is what a hallucination actually is; this isn't
a second, different hallucination-detection mechanism.

Cost/latency tradeoff, accepted deliberately: this adds one more LLM call
(gpt-4o-mini) to every live query, since Faithfulness itself needs an LLM
to judge groundedness against the context. A real production system at
higher volume would likely sample (score every Nth request) rather than
score every single one — documented here as a real Scalability
consideration, not silently ignored.

This is DETECTION only, not correction: it flags the response, it doesn't
retry retrieval or regenerate the answer. Wiring in an automatic
re-retrieve-on-failed-check loop is the (larger, separate) Self-Reflective
RAG upgrade — this guardrail is a real, useful step on the way there, not
a replacement for it.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

logger = logging.getLogger(__name__)

HALLUCINATION_THRESHOLD = 0.7
_GUARDRAIL_MODEL = "gpt-4o-mini"

# Built lazily, not at import time: constructing ChatOpenAI requires
# OPENAI_API_KEY to already be loaded, and importing this module shouldn't
# have a hard dependency on that (e.g. during offline test collection).
_faithfulness_metric: Optional[Faithfulness] = None


def _get_metric() -> Faithfulness:
    global _faithfulness_metric
    if _faithfulness_metric is None:
        llm = LangchainLLMWrapper(ChatOpenAI(model=_GUARDRAIL_MODEL, temperature=0))
        _faithfulness_metric = Faithfulness(llm=llm)
    return _faithfulness_metric


def score_faithfulness(query: str, answer: str, contexts: List[str]) -> Optional[float]:
    """Returns the Faithfulness score (0-1), or None if scoring itself
    couldn't run (no context to score against, or the scoring call
    failed) — a guardrail that can't verify should report "unknown," not
    silently claim the answer is fine."""
    if not contexts:
        return None
    try:
        sample = SingleTurnSample(user_input=query, retrieved_contexts=contexts, response=answer)
        return _get_metric().single_turn_score(sample)
    except Exception:
        logger.exception("hallucination_guardrail_failed")
        return None


def is_likely_hallucination(score: Optional[float]) -> bool:
    return score is not None and score < HALLUCINATION_THRESHOLD
