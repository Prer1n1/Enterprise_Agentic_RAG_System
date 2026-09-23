"""Before/after benchmark for the Laya classification pilot (see
laya_classifier.py, ingestion/metadata_extractor.py's use_laya param).

Deliberately separate from evaluation/ragas_eval.py: that harness scores
the end-to-end RAG pipeline (routing/retrieval/reranking/generation) over
QUERIES. This one scores a single, narrower decision — document category
classification at INGESTION — over hand-labeled EXCERPTS, which is the
only decision point this pilot actually touches (see laya_classifier.py's
docstring for why classification, not synthesis or routing, was chosen).

Hand-curated on purpose, same reasoning as evaluation/eval_dataset.py: a
handful of unambiguous, single-category excerpts is enough to compare
three classifiers' accuracy/latency without paying for a large labeled
set that wouldn't add much signal at this scope.

Three tiers compared, each in isolation (not the real fallback chain):
  keyword  -> classify_category(use_llm=False, use_laya=False)
  llm      -> classify_category(use_llm=True,  use_laya=False)  [existing baseline]
  laya     -> classify_category(use_llm=False, use_laya=True)   [pilot, alone —
              use_llm=False here so a Laya failure shows up as a miss instead
              of silently being rescued by the LLM tier, which would hide
              the pilot's own real accuracy]

Note: this actually loads and runs the real local Laya model — no API key
needed (unlike the earlier, abandoned Jev version of this benchmark), but
the first run downloads the checkpoint (~1.6GB) and every run pays a real
model-load cost (~20-25s measured). Not something to run on every CI push
for the same reason test_laya_classifier.py's live test isn't CI-wired —
see that file's docstring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import config  # noqa: F401 — importing this loads .env (OPENAI_API_KEY) as a
# side effect; without it the LLM tier below silently fails on every call
# (openai.OpenAIError: Missing credentials) and classify_category() falls
# through to the keyword classifier without raising — a real bug this
# project hit while building this exact script, caught only by noticing
# the LLM tier's latency was suspiciously ~0ms instead of a real network
# round-trip. See docs/design-decisions.md.
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


def _run_tier(name: str, use_llm: bool, use_laya: bool) -> None:
    correct = 0
    total_seconds = 0.0
    print(f"\n--- {name} ---")
    for case in CASES:
        start = time.perf_counter()
        predicted = classify_category(case.excerpt, use_llm=use_llm, use_laya=use_laya)
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
    print("=== Classification benchmark: keyword vs LLM vs Laya (pilot) ===")
    print(f"{len(CASES)} hand-labeled excerpts, one per category.")
    print("Loading Laya (first run downloads the checkpoint; every run pays a real load cost)...")

    _run_tier("keyword (existing fallback tier)", use_llm=False, use_laya=False)
    _run_tier("LLM — gpt-4o-mini (existing primary tier)", use_llm=True, use_laya=False)
    _run_tier("Laya (pilot, alone — no LLM fallback)", use_llm=False, use_laya=True)


if __name__ == "__main__":
    main()
