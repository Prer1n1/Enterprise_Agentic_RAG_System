"""Curated evaluation set — hand-picked (question, reference_answer)
pairs grounded in sample_data's actual content. Kept small and
hand-curated on purpose: every RAGAS metric here is an LLM judgment call
per sample, so a bigger set costs real API spend without adding much
signal at this corpus size (10-12 chunks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    question: str
    reference_answer: str  # ground truth, taken directly from the source documents
    # Which category/categories a CORRECT routing decision must include at
    # least one of — used for the Routing Accuracy metric. Intersection-
    # based, not exact-match: the router is deliberately allowed to be
    # broader than strictly necessary (see agent/planner.py's own
    # "more for a question that spans domains" guidance), so a routing
    # decision that includes an expected category PLUS extras is still
    # correct. Only a routing decision that misses every expected
    # category entirely counts as a routing failure.
    expected_categories: List[str]


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        question="How many paid leave days do employees get per year?",
        reference_answer="Employees are entitled to 20 days of paid leave per year.",
        expected_categories=["HR"],
    ),
    EvalCase(
        question="How often must passwords be rotated?",
        reference_answer="Passwords must be rotated every 90 days.",
        expected_categories=["IT", "Security"],
    ),
    EvalCase(
        question="How quickly must security incidents be reported?",
        reference_answer="Security incidents must be reported within 1 hour of discovery.",
        expected_categories=["Security"],
    ),
    EvalCase(
        question="What is the expense approval threshold?",
        reference_answer="Expenses over $500 require manager approval.",
        expected_categories=["Finance"],
    ),
    EvalCase(
        question="How long does it take to get a laptop as a new employee?",
        reference_answer="A laptop will be provisioned within 2 business days.",
        expected_categories=["IT", "HR"],
    ),
]
