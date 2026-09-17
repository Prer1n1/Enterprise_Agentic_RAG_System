"""End-to-end pipeline sanity check, using a generic fake embedding model
so it runs free/offline. Confirms: first run ingests everything, second
run (no changes) skips everything via the tracker."""

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
result1 = ingest_directory(SAMPLE_DIR, tracker=tracker, embeddings=fake_embeddings)
print(f"ingested_files: {len(result1.ingested_files)}")
print(f"skipped_unchanged: {len(result1.skipped_unchanged)}")
print(f"chunks produced: {len(result1.chunks)}")
assert len(result1.ingested_files) == 4
assert len(result1.skipped_unchanged) == 0
assert len(result1.chunks) > 0

print("\n=== Run 2: same files, nothing changed ===")
result2 = ingest_directory(SAMPLE_DIR, tracker=tracker, embeddings=fake_embeddings)
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
result_a = ingest_directory(root_a, tracker=shared_tracker, embeddings=fake_embeddings)
print(f"ingest root_a: ingested={len(result_a.ingested_files)} deleted={len(result_a.deleted_files)}")
assert len(result_a.ingested_files) == 1
assert len(result_a.deleted_files) == 0

result_b = ingest_directory(root_b, tracker=shared_tracker, embeddings=fake_embeddings)
print(f"ingest root_b: ingested={len(result_b.ingested_files)} deleted={len(result_b.deleted_files)}")
assert len(result_b.ingested_files) == 1
assert len(result_b.deleted_files) == 0, "root_a's file must NOT be reported as deleted just because it's absent from root_b's listing"
print("PASS: ingesting root_b did not report root_a's file as deleted")

shutil.rmtree(tmp_dir)
print("\nALL PIPELINE TESTS PASSED — run 2 correctly did zero work")
