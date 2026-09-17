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


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        question="How many paid leave days do employees get per year?",
        reference_answer="Employees are entitled to 20 days of paid leave per year.",
    ),
    EvalCase(
        question="How often must passwords be rotated?",
        reference_answer="Passwords must be rotated every 90 days.",
    ),
    EvalCase(
        question="How quickly must security incidents be reported?",
        reference_answer="Security incidents must be reported within 1 hour of discovery.",
    ),
    EvalCase(
        question="What is the expense approval threshold?",
        reference_answer="Expenses over $500 require manager approval.",
    ),
    EvalCase(
        question="How long does it take to get a laptop as a new employee?",
        reference_answer="A laptop will be provisioned within 2 business days.",
    ),
]
