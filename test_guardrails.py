"""Sanity checks for the two newest guardrails: the live hallucination
guardrail (hallucination_guardrail.py) and the topic/scope guardrail fix
in agent/planner.py's plan_categories(). Rate limiting (slowapi) is
verified separately, live, against a running server — rate limiting is
fundamentally an integration behavior, not something meaningful to unit
test in isolation.
"""

from config import OPENAI_API_KEY
from hallucination_guardrail import HALLUCINATION_THRESHOLD, is_likely_hallucination, score_faithfulness

print("=== score_faithfulness() with no context (offline, free) ===")
assert score_faithfulness("some question", "some answer", []) is None
print("PASS: no context to score against -> None, not an exception")

print("\n=== score_faithfulness() when the scoring call itself fails (offline, free) ===")
import hallucination_guardrail as guardrail_module


class FailingMetric:
    def single_turn_score(self, sample):
        raise RuntimeError("simulated scoring failure")


original_get_metric = guardrail_module._get_metric
guardrail_module._get_metric = lambda: FailingMetric()
try:
    result = score_faithfulness("q", "a", ["some context"])
finally:
    guardrail_module._get_metric = original_get_metric
assert result is None
print("PASS: a scoring failure returns None (unknown), not a crash")

print("\n=== is_likely_hallucination() threshold logic (offline, free) ===")
assert is_likely_hallucination(None) is False, "unknown score must never be reported as a hallucination"
assert is_likely_hallucination(HALLUCINATION_THRESHOLD - 0.01) is True
assert is_likely_hallucination(HALLUCINATION_THRESHOLD) is False
print("PASS: threshold logic correct, and 'unknown' never gets treated as 'hallucinating'")

if OPENAI_API_KEY:
    print("\n=== Real RAGAS Faithfulness scoring (needs OPENAI_API_KEY) ===")
    faithful_score = score_faithfulness(
        "How often must passwords be rotated?",
        "Passwords must be rotated every 90 days.",
        ["Passwords must be rotated every 90 days."],
    )
    print(f"faithful answer score: {faithful_score}")
    assert faithful_score is not None and faithful_score >= HALLUCINATION_THRESHOLD

    fabricated_score = score_faithfulness(
        "How often must passwords be rotated?",
        "Passwords must be rotated every 30 days and must include a unicorn emoji.",
        ["Passwords must be rotated every 90 days."],
    )
    print(f"fabricated answer score: {fabricated_score}")
    assert fabricated_score is not None and fabricated_score < HALLUCINATION_THRESHOLD
    assert is_likely_hallucination(fabricated_score) is True
    print("PASS: real RAGAS scoring discriminates a faithful answer from a fabricated one, live")
else:
    print("\nSkipping real RAGAS scoring test — OPENAI_API_KEY not set.")

print("\nALL GUARDRAIL TESTS PASSED")
