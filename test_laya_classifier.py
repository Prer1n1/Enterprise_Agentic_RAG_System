"""Sanity checks for laya_classifier.py and its wiring into
classify_category(). Runs free/offline: the singleton-load-failure
fallback, the confidence gate, and the opt-in-only default are all proven
by monkeypatching laya_classifier's internals rather than actually loading
the real model — that would mean downloading ~1.6GB of weights and
~20-second load times on every CI run, which is a real cost this test
suite deliberately avoids paying on every push. See the bottom of this
file for the separate, NOT-CI-wired live test that actually loads and
runs the real model.
"""

import laya_classifier as laya_module
from ingestion.metadata_extractor import CATEGORIES, _LAYA_CRITERIA, classify_category

print("=== laya_classify_category() when the model fails to load (monkeypatched) ===")
original_get_agent = laya_module._get_agent
laya_module._get_agent = lambda: None
try:
    result = laya_module.laya_classify_category("some HR policy text", CATEGORIES, _LAYA_CRITERIA)
finally:
    laya_module._get_agent = original_get_agent
assert result is None, "laya_classify_category() must return None (not raise) when the model is unavailable"
print("PASS: laya_classify_category() returns None gracefully when the model can't be loaded")


class FakeAgent:
    def __init__(self, response):
        self._response = response

    def predict(self, text, questions):
        return self._response


print("\n=== laya_classify_category() rejects a low-confidence answer ===")
laya_module._get_agent = lambda: FakeAgent(
    {"answers": {"category": {"choice": "HR", "confidence": 0.1}}}
)
try:
    result = laya_module.laya_classify_category("ambiguous text", CATEGORIES, _LAYA_CRITERIA)
finally:
    laya_module._get_agent = original_get_agent
assert result is None, f"a 0.1-confidence answer must be rejected (below {laya_module.LAYA_CONFIDENCE_THRESHOLD}), got {result}"
print("PASS: low-confidence Laya answers are rejected, not trusted")

print("\n=== laya_classify_category() rejects a malformed response instead of raising ===")
laya_module._get_agent = lambda: FakeAgent({"unexpected": "shape"})
try:
    result = laya_module.laya_classify_category("some text", CATEGORIES, _LAYA_CRITERIA)
finally:
    laya_module._get_agent = original_get_agent
assert result is None, "a malformed response must degrade to None, never raise"
print("PASS: malformed Laya responses degrade to None instead of raising")

print("\n=== laya_classify_category() returns a confident, valid answer ===")


class RecordingAgent:
    def predict(self, text, questions):
        assert questions["category"]["type"] == "choice"
        assert set(questions["category"]["criteria"].keys()) == set(CATEGORIES)
        return {"answers": {"category": {"choice": "Security", "confidence": 0.94}}}


laya_module._get_agent = lambda: RecordingAgent()
try:
    result = laya_module.laya_classify_category("password rotation policy", CATEGORIES, _LAYA_CRITERIA)
finally:
    laya_module._get_agent = original_get_agent
assert result == ("Security", 0.94), f"expected ('Security', 0.94), got {result}"
print("PASS: a confident, valid Laya answer is parsed and returned correctly")

print("\n=== classify_category(use_laya=True) actually tries Laya first, then falls back ===")
laya_module._get_agent = lambda: RecordingAgent()
try:
    category = classify_category("password rotation policy", use_llm=False, use_laya=True)
finally:
    laya_module._get_agent = original_get_agent
assert category == "Security", f"expected Laya's answer 'Security' to win, got {category}"
print("PASS: classify_category(use_laya=True) uses Laya's answer ahead of the keyword fallback")

laya_module._get_agent = lambda: None
try:
    category = classify_category("password rotation policy every 90 days", use_llm=False, use_laya=True)
finally:
    laya_module._get_agent = original_get_agent
assert category == "Security", (
    f"with Laya unavailable, must fall through to the keyword classifier "
    f"(which should still find 'password' -> Security), got {category}"
)
print("PASS: classify_category(use_laya=True) falls all the way through to keywords when Laya is unavailable")

print("\n=== classify_category(use_laya=False) never touches Laya at all (default, opt-in only) ===")


class ShouldNotBeCalledAgent:
    def predict(self, text, questions):
        raise AssertionError("Laya must not be called when use_laya=False")


laya_module._get_agent = lambda: ShouldNotBeCalledAgent()
try:
    category = classify_category("some HR text about leave", use_llm=False, use_laya=False)
finally:
    laya_module._get_agent = original_get_agent
print("PASS: use_laya defaults to False and never invokes Laya unless explicitly opted in")

print("\nALL LAYA CLASSIFIER TESTS PASSED (offline)")

# --- Live test, deliberately NOT wired into CI ---
# Actually downloads (~1.6GB, first run only) and loads (~24s, every run)
# the real convaiinnovations/laya checkpoint and runs real inference.
# Opt-in via RUN_LAYA_LIVE_TEST=1 -- not added to .github/workflows/tests.yml
# because paying a multi-minute download + load cost on every push/PR would
# be a real, ongoing tax on this project's CI time for a pilot integration,
# unlike the free/offline checks above. Run manually with:
#   RUN_LAYA_LIVE_TEST=1 python test_laya_classifier.py
import os

if os.environ.get("RUN_LAYA_LIVE_TEST") == "1":
    print("\n=== Live test: real Laya model, real inference (RUN_LAYA_LIVE_TEST=1) ===")
    text = "Passwords must be rotated every 90 days and must include a special character."
    result = laya_module.laya_classify_category(text, CATEGORIES, _LAYA_CRITERIA)
    assert result is not None, "a real Laya call should not fail"
    category, confidence = result
    print(f"Laya classified as: {category!r} (confidence={confidence:.3f})")
    assert category == "Security", f"expected 'Security' for a password-rotation policy, got {category!r}"
    print("PASS: real Laya inference correctly classified a password-rotation policy as Security")
else:
    print("\nSkipping live test — set RUN_LAYA_LIVE_TEST=1 to actually load and run the real model.")
