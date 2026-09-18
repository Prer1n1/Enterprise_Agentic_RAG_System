"""End-to-end pipeline sanity check, using a generic fake embedding model
and use_llm_classifier=False (forces the free keyword-based category
fallback, not a real LLM call) so it runs free/offline. Confirms: first run
ingests everything, second run (no changes) skips everything via the
tracker. The real LLM classification path is verified separately in
test_metadata.py, gated behind OPENAI_API_KEY.

ingest_directory() no longer calls tracker.mark_ingested() itself — see
docs/design-decisions.md ("Reliability / Data Integrity") for the real bug
that fixed. The caller now marks each file in result.pending_mark AFTER
persisting its chunks, exactly as main.py/api/app.py do — this test
mirrors that same two-step sequence rather than relying on
ingest_directory() to do it internally."""

import hashlib
import shutil
import tempfile
from pathlib import Path

from ingestion.pipeline import ingest_directory
from ingestion.tracker import IngestionTracker

SAMPLE_DIR = Path(__file__).parent / "sample_data"


class HashFakeEmbeddings:
    """Deterministic fake vector per text, works for ANY input (unlike the
    fixed-mapping FakeEmbeddings in test_chunking.py) — good enough to
    prove the pipeline WIRES TOGETHER correctly, not to judge chunk quality."""

    def embed_documents(self, texts):
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vectors.append([b / 255 for b in h[:8]])
        return vectors


tmp_dir = Path(tempfile.mkdtemp())
db_path = tmp_dir / "manifest.db"
tracker = IngestionTracker(db_path=db_path)
fake_embeddings = HashFakeEmbeddings()

print("=== Run 1: first ingestion, all files are new ===")
result1 = ingest_directory(SAMPLE_DIR, tracker=tracker, embeddings=fake_embeddings, use_llm_classifier=False)
print(f"ingested_files: {len(result1.ingested_files)}")
print(f"skipped_unchanged: {len(result1.skipped_unchanged)}")
print(f"chunks produced: {len(result1.chunks)}")
assert len(result1.ingested_files) == 4
assert len(result1.skipped_unchanged) == 0
assert len(result1.chunks) > 0
assert len(result1.pending_mark) == 4

# Simulates the caller persisting chunks THEN marking — the real sequence
# main.py/api/app.py now follow.
for file_path in result1.pending_mark:
    tracker.mark_ingested(file_path)

print("\n=== Run 2: same files, nothing changed ===")
result2 = ingest_directory(SAMPLE_DIR, tracker=tracker, embeddings=fake_embeddings, use_llm_classifier=False)
print(f"ingested_files: {len(result2.ingested_files)}")
print(f"skipped_unchanged: {len(result2.skipped_unchanged)}")
print(f"chunks produced: {len(result2.chunks)}")
assert len(result2.ingested_files) == 0
assert len(result2.skipped_unchanged) == 4
assert len(result2.chunks) == 0

print("\n=== Regression: two separate ingestion roots sharing one tracker ===")
print("(real bug found via the /documents/upload endpoint: ingesting one root")
print(" wrongly reported every file in the OTHER root as 'deleted')")
root_a = tmp_dir / "root_a"
root_b = tmp_dir / "root_b"
root_a.mkdir()
root_b.mkdir()
(root_a / "a.csv").write_text("policy_id,department,rule\nP-1,HR,Some HR rule\n")
(root_b / "b.csv").write_text("policy_id,department,rule\nP-2,IT,Some IT rule\n")

shared_tracker = IngestionTracker(db_path=tmp_dir / "shared_manifest.db")
result_a = ingest_directory(root_a, tracker=shared_tracker, embeddings=fake_embeddings, use_llm_classifier=False)
print(f"ingest root_a: ingested={len(result_a.ingested_files)} deleted={len(result_a.deleted_files)}")
assert len(result_a.ingested_files) == 1
assert len(result_a.deleted_files) == 0
for file_path in result_a.pending_mark:
    shared_tracker.mark_ingested(file_path)

result_b = ingest_directory(root_b, tracker=shared_tracker, embeddings=fake_embeddings, use_llm_classifier=False)
print(f"ingest root_b: ingested={len(result_b.ingested_files)} deleted={len(result_b.deleted_files)}")
assert len(result_b.ingested_files) == 1
assert len(result_b.deleted_files) == 0, "root_a's file must NOT be reported as deleted just because it's absent from root_b's listing"
for file_path in result_b.pending_mark:
    shared_tracker.mark_ingested(file_path)
print("PASS: ingesting root_b did not report root_a's file as deleted")

print("\n=== Regression: a file must NOT be marked ingested until its chunks are")
print("    actually persisted — not just successfully chunked ===")
print("(real bug: mark_ingested() used to run INSIDE ingest_directory(), before")
print(" the caller ever persisted result.chunks — a crash or failed save between")
print(" the two would leave the tracker believing a file was ingested when its")
print(" chunks were never actually stored anywhere, and the next run would skip")
print(" it forever)")
crash_dir = tmp_dir / "crash_sim"
crash_dir.mkdir()
(crash_dir / "c.csv").write_text("policy_id,department,rule\nP-3,Finance,Some Finance rule\n")
crash_tracker = IngestionTracker(db_path=tmp_dir / "crash_manifest.db")

result_c = ingest_directory(crash_dir, tracker=crash_tracker, embeddings=fake_embeddings, use_llm_classifier=False)
assert len(result_c.pending_mark) == 1
# Simulate persistence FAILING here (e.g. a crash between chunking and
# save_chunks()/add_chunks()) — deliberately do NOT call mark_ingested().

decision = crash_tracker.check(result_c.pending_mark[0])
assert decision.should_ingest, (
    "a file whose chunks were never persisted must still be reported as "
    "needing ingestion on the next run — mark_ingested() must not have "
    "been called for it"
)
print(f"PASS: unmarked file still reports should_ingest=True (reason={decision.reason!r}) — nothing silently lost")

shutil.rmtree(tmp_dir)
print("\nALL PIPELINE TESTS PASSED — run 2 correctly did zero work")
