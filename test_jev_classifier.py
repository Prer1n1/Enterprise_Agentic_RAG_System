"""Sanity checks for jev_classifier.py and its wiring into
classify_category(). Runs free/offline: the no-TYPESAFE_API_KEY fallback,
a low-confidence fallback, and a malformed-response fallback are all
proven by monkeypatching jev_classifier's internals rather than depending
on whether a real key happens to be configured locally — same "must pass
identically either way" discipline as test_reranker.py.
"""

import jev_classifier as jev_module
from config import TYPESAFE_API_KEY
from ingestion.metadata_extractor import CATEGORIES, _JEV_CRITERIA, classify_category

print("=== jev_classify_category() with no TYPESAFE_API_KEY configured (monkeypatched) ===")
original_key = jev_module.TYPESAFE_API_KEY
jev_module.TYPESAFE_API_KEY = None
try:
    result = jev_module.jev_classify_category("some HR policy text", CATEGORIES, _JEV_CRITERIA)
finally:
    jev_module.TYPESAFE_API_KEY = original_key
assert result is None, "jev_classify_category() must return None (not raise) when no key is configured"
print("PASS: jev_classify_category() returns None gracefully with no key")

print("\n=== jev_classify_category() rejects a low-confidence answer ===")
jev_module.TYPESAFE_API_KEY = "fake-key-for-this-test"


def low_confidence_call(payload):
    return {"answers": {"category": {"choice": "HR", "confidence": 0.2}}}


original_call = jev_module._call_jev
jev_module._call_jev = low_confidence_call
try:
    result = jev_module.jev_classify_category("ambiguous text", CATEGORIES, _JEV_CRITERIA)
finally:
    jev_module._call_jev = original_call
    jev_module.TYPESAFE_API_KEY = original_key
assert result is None, f"a 0.2-confidence answer must be rejected (below {jev_module.JEV_CONFIDENCE_THRESHOLD}), got {result}"
print("PASS: low-confidence Jev answers are rejected, not trusted")

print("\n=== jev_classify_category() rejects a malformed response instead of raising ===")
jev_module.TYPESAFE_API_KEY = "fake-key-for-this-test"


def malformed_call(payload):
    return {"unexpected": "shape"}


jev_module._call_jev = malformed_call
try:
    result = jev_module.jev_classify_category("some text", CATEGORIES, _JEV_CRITERIA)
finally:
    jev_module._call_jev = original_call
    jev_module.TYPESAFE_API_KEY = original_key
assert result is None, "a malformed response must degrade to None, never raise"
print("PASS: malformed Jev responses degrade to None instead of raising")

print("\n=== jev_classify_category() returns a confident, valid answer ===")
jev_module.TYPESAFE_API_KEY = "fake-key-for-this-test"


def confident_call(payload):
    assert payload["questions"]["category"]["type"] == "choice"
    assert set(payload["questions"]["category"]["criteria"].keys()) == set(CATEGORIES)
    return {"answers": {"category": {"choice": "Security", "confidence": 0.94}}}


jev_module._call_jev = confident_call
try:
    result = jev_module.jev_classify_category("password rotation policy", CATEGORIES, _JEV_CRITERIA)
finally:
    jev_module._call_jev = original_call
    jev_module.TYPESAFE_API_KEY = original_key
assert result == ("Security", 0.94), f"expected ('Security', 0.94), got {result}"
print("PASS: a confident, valid Jev answer is parsed and returned correctly")

print("\n=== classify_category(use_jev=True) actually tries Jev first, then falls back ===")
import ingestion.metadata_extractor as metadata_module

jev_module.TYPESAFE_API_KEY = "fake-key-for-this-test"
jev_module._call_jev = confident_call
try:
    category = classify_category("password rotation policy", use_llm=False, use_jev=True)
finally:
    jev_module._call_jev = original_call
    jev_module.TYPESAFE_API_KEY = original_key
assert category == "Security", f"expected Jev's answer 'Security' to win, got {category}"
print("PASS: classify_category(use_jev=True) uses Jev's answer ahead of the keyword fallback")

jev_module.TYPESAFE_API_KEY = None
category = classify_category("password rotation policy every 90 days", use_llm=False, use_jev=True)
assert category == "Security", (
    f"with Jev unavailable, must fall through to the keyword classifier "
    f"(which should still find 'password' -> Security), got {category}"
)
jev_module.TYPESAFE_API_KEY = original_key
print("PASS: classify_category(use_jev=True) falls all the way through to keywords when Jev is unavailable")

print("\n=== classify_category(use_jev=False) never calls Jev at all (default, opt-in only) ===")


def should_not_be_called(payload):
    raise AssertionError("Jev must not be called when use_jev=False")


jev_module.TYPESAFE_API_KEY = "fake-key-for-this-test"
jev_module._call_jev = should_not_be_called
try:
    category = classify_category("some HR text about leave", use_llm=False, use_jev=False)
finally:
    jev_module._call_jev = original_call
    jev_module.TYPESAFE_API_KEY = original_key
print("PASS: use_jev defaults to False and never invokes Jev unless explicitly opted in")

if TYPESAFE_API_KEY:
    print("\n=== Part 2: real Jev API call (needs TYPESAFE_API_KEY) ===")
    text = "Passwords must be rotated every 90 days and must include a special character."
    result = jev_module.jev_classify_category(text, CATEGORIES, _JEV_CRITERIA)
    assert result is not None, "a real Jev call with a valid key should not fail"
    category, confidence = result
    print(f"Jev classified as: {category!r} (confidence={confidence:.2f})")
    assert category == "Security", f"expected 'Security' for a password-rotation policy, got {category!r}"
    print("PASS: real Jev call correctly classified a password-rotation policy as Security")
else:
    print("\nSkipping Part 2 (real Jev API test) — TYPESAFE_API_KEY not set.")

print("\nALL JEV CLASSIFIER TESTS PASSED")
