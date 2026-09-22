"""Before/after benchmark for the Jev classification pilot (see
jev_classifier.py, ingestion/metadata_extractor.py's use_jev param).

Deliberately separate from evaluation/ragas_eval.py: that harness scores
the end-to-end RAG pipeline (routing/retrieval/reranking/generation) over
QUERIES. This one scores a single, narrower decision — document category
classification at INGESTION — over hand-labeled EXCERPTS, which is the
only decision point this pilot actually touches (see jev_classifier.py's
docstring for why classification, not synthesis or routing, was chosen).

Hand-curated on purpose, same reasoning as evaluation/eval_dataset.py: a
handful of unambiguous, single-category excerpts is enough to compare
three classifiers' accuracy/latency without paying for a large labeled
set that wouldn't add much signal at this scope.

Three tiers compared, each in isolation (not the real fallback chain):
  keyword  -> classify_category(use_llm=False, use_jev=False)
  llm      -> classify_category(use_llm=True,  use_jev=False)   [existing baseline]
  jev      -> classify_category(use_llm=False, use_jev=True)    [pilot, alone —
              use_llm=False here so a Jev failure shows up as a miss instead
              of silently being rescued by the LLM tier, which would hide
              the pilot's own real accuracy]

If TYPESAFE_API_KEY isn't set, the "jev" column is skipped entirely rather
than reported as if Jev had been tested — printing use_jev=True with no
key configured would just re-run the keyword fallback and silently produce
fabricated "Jev" numbers that were never really Jev's answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from config import TYPESAFE_API_KEY
from ingestion.metadata_extractor import classify_category


@dataclass
class ClassificationCase:
    excerpt: str
    expected_category: str


CASES: List[ClassificationCase] = [
    ClassificationCase(
        "Employees are entitled to 20 days of paid leave per year. New hires "
        "must complete onboarding paperwork with HR within their first week, "
        "including benefits enrollment.",
        "HR",
    ),
    ClassificationCase(
        "All expenses over $500 require manager approval before reimbursement. "
        "Submit invoices through the finance portal along with your budget code.",
        "Finance",
    ),
    ClassificationCase(
        "Passwords must be rotated every 90 days and must include a special "
        "character. Security incidents must be reported within 1 hour of "
        "discovery to the security team.",
        "Security",
    ),
    ClassificationCase(
        "New employee laptops are provisioned within 2 business days by IT. "
        "All systems must be deployed through the approved infrastructure "
        "pipeline and connected via the corporate VPN.",
        "IT",
    ),
    ClassificationCase(
        "This agreement is governed by the terms outlined below. Both parties "
        "accept liability limitations as described in the compliance section "
        "of this contract.",
        "Legal",
    ),
    ClassificationCase(
        "The office kitchen is stocked with coffee and snacks every Monday. "
        "Parking passes are available at the front desk.",
        "General",
    ),
]


def _run_tier(name: str, use_llm: bool, use_jev: bool) -> None:
    correct = 0
    total_seconds = 0.0
    print(f"\n--- {name} ---")
    for case in CASES:
        start = time.perf_counter()
        predicted = classify_category(case.excerpt, use_llm=use_llm, use_jev=use_jev)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        ok = predicted == case.expected_category
        correct += ok
        print(
            f"  expected={case.expected_category:<10} predicted={predicted:<10} "
            f"{'OK' if ok else 'MISS':<4} ({elapsed * 1000:.0f}ms)"
        )
    n = len(CASES)
    print(f"  accuracy: {correct}/{n} ({100 * correct / n:.0f}%)   avg latency: {1000 * total_seconds / n:.0f}ms")


def main() -> None:
    print("=== Classification benchmark: keyword vs LLM vs Jev (pilot) ===")
    print(f"{len(CASES)} hand-labeled excerpts, one per category.\n")

    _run_tier("keyword (existing fallback tier)", use_llm=False, use_jev=False)
    _run_tier("LLM — gpt-4o-mini (existing primary tier)", use_llm=True, use_jev=False)

    if TYPESAFE_API_KEY:
        _run_tier("Jev (pilot, alone — no LLM fallback)", use_llm=False, use_jev=True)
    else:
        print(
            "\n--- Jev (pilot) ---\n"
            "  SKIPPED — no TYPESAFE_API_KEY configured. Running use_jev=True "
            "without a key would just silently re-run the keyword fallback and "
            "produce numbers that were never really Jev's answers, so this "
            "benchmark refuses to report a 'Jev' row until a real key is set."
        )


if __name__ == "__main__":
    main()
