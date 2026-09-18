"""Sanity checks for pii_redaction.py and its wiring into
ingest_directory().

Part 1 (free/offline) uses a FAKE structured-output LLM double — there's
no keyword fallback to fall back to here (see pii_redaction.py's module
docstring for why: fail-closed, not fail-open), so offline testing means
injecting a fake `llm` object rather than passing use_llm=False like the
classifier/injection detector.

Part 2 (gated behind OPENAI_API_KEY) proves the real LLM detector finds
and correctly redacts real PII (SSN, email, salary) while leaving benign
text untouched.
"""

import shutil
import tempfile
from pathlib import Path

from config import OPENAI_API_KEY
from ingestion.pipeline import ingest_directory
from ingestion.tracker import IngestionTracker
from pii_redaction import PIIDetectionFailed, PIIDetectionResult, PIIEntity, redact_pii


class FakeStructuredLLM:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, prompt):
        if self._error:
            raise self._error
        return self._result


class FakeLLM:
    def __init__(self, result=None, error=None):
        self._structured = FakeStructuredLLM(result=result, error=error)

    def with_structured_output(self, schema):
        return self._structured


class HashFakeEmbeddings:
    def embed_documents(self, texts):
        import hashlib

        return [[b / 255 for b in hashlib.sha256(t.encode("utf-8")).digest()[:8]] for t in texts]


print("=== Part 1: redact_pii() with a fake LLM (offline, free) ===")
fake_result = PIIDetectionResult(
    entities=[
        PIIEntity(text="123-45-6789", type="SSN"),
        PIIEntity(text="jane@example.com", type="EMAIL"),
    ]
)
redacted, types_found = redact_pii(
    "Employee SSN is 123-45-6789, contact jane@example.com for questions.",
    llm=FakeLLM(result=fake_result),
)
print(f"redacted: {redacted!r}")
assert "123-45-6789" not in redacted
assert "jane@example.com" not in redacted
assert "[SSN_REDACTED]" in redacted
assert "[EMAIL_REDACTED]" in redacted
assert sorted(types_found) == ["EMAIL", "SSN"]
print("PASS: both entities replaced with type-tagged placeholders")

print("\n=== Part 1b: redact_pii() with zero entities found (no-op) ===")
redacted, types_found = redact_pii(
    "All expenses over $500 require manager approval.",
    llm=FakeLLM(result=PIIDetectionResult(entities=[])),
)
assert redacted == "All expenses over $500 require manager approval."
assert types_found == []
print("PASS: text unchanged when nothing is flagged")

print("\n=== Part 1c: redact_pii() fails CLOSED when the LLM call fails ===")
try:
    redact_pii("some text", llm=FakeLLM(error=ValueError("simulated API failure")))
    print("FAIL: should have raised PIIDetectionFailed")
    raise SystemExit(1)
except PIIDetectionFailed:
    print("PASS: PIIDetectionFailed raised — no silent fallback to storing unredacted text")

print("\n=== Part 1d: ingest_directory() redacts PII before chunking (offline) ===")
tmp_dir = Path(tempfile.mkdtemp())
src_dir = tmp_dir / "src"
src_dir.mkdir()
(src_dir / "employee.csv").write_text(
    "policy_id,department,rule\nP-1,HR,Employee SSN on file is 123-45-6789 for payroll purposes\n"
)

tracker = IngestionTracker(db_path=tmp_dir / "manifest.db")
pii_fake = FakeLLM(result=PIIDetectionResult(entities=[PIIEntity(text="123-45-6789", type="SSN")]))
result = ingest_directory(
    src_dir,
    tracker=tracker,
    embeddings=HashFakeEmbeddings(),
    use_llm_classifier=False,
    use_llm_injection_detector=False,
    pii_llm=pii_fake,
)
assert len(result.chunks) > 0
all_chunk_text = " ".join(c.content for c in result.chunks)
assert "123-45-6789" not in all_chunk_text, "raw SSN must never reach the chunked/stored content"
assert "[SSN_REDACTED]" in all_chunk_text
print("PASS: the SSN was redacted before chunking — stored content has the placeholder, not the raw value")

print("\n=== Part 1e: ingest_directory() blocks a file when the PII check itself fails (offline) ===")
failing_pii_llm = FakeLLM(error=ValueError("simulated API failure"))
result2 = ingest_directory(
    src_dir,
    tracker=IngestionTracker(db_path=tmp_dir / "manifest2.db"),
    embeddings=HashFakeEmbeddings(),
    use_llm_classifier=False,
    use_llm_injection_detector=False,
    pii_llm=failing_pii_llm,
)
assert len(result2.pii_check_failed) == 1 and "employee.csv" in result2.pii_check_failed[0]
assert len(result2.chunks) == 0
assert len(result2.ingested_files) == 0
print("PASS: a PII-check failure blocks the file (fail-closed), reported in pii_check_failed")

shutil.rmtree(tmp_dir, ignore_errors=True)

if OPENAI_API_KEY:
    print("\n=== Part 2: real LLM detector (needs OPENAI_API_KEY) ===")
    text = (
        "Employee John Doe's SSN is 987-65-4320 and his annual salary is $95,000. "
        "Contact him at john.doe@example.com or (555) 123-4567. "
        "All expenses over $500 require manager approval before reimbursement."
    )
    redacted, types_found = redact_pii(text)
    print(f"types found: {sorted(set(types_found))}")
    print(f"redacted:\n{redacted}")

    assert "987-65-4320" not in redacted, "SSN must be redacted"
    assert "john.doe@example.com" not in redacted, "email must be redacted"
    assert "95,000" not in redacted, "salary figure must be redacted"
    assert "555" not in redacted or "PHONE_NUMBER_REDACTED" in redacted, "phone number must be redacted"
    assert "expenses over $500 require manager approval" in redacted, (
        "ordinary policy text unrelated to a specific person's PII must survive untouched"
    )
    print("PASS: real LLM detector redacted SSN/email/salary/phone while leaving generic policy text intact")
else:
    print("\nSkipping Part 2 (real LLM detector test) — OPENAI_API_KEY not set.")

print("\nALL PII REDACTION TESTS PASSED")
