"""Sanity checks for the incremental ingestion tracker. Uses temp files +
a temp SQLite DB so it never touches the real project manifest."""

import shutil
import tempfile
from pathlib import Path

from ingestion.schema import canonical_source
from ingestion.tracker import IngestionTracker

tmp_dir = Path(tempfile.mkdtemp())
db_path = tmp_dir / "test_manifest.db"

file_a = tmp_dir / "a.txt"
file_b = tmp_dir / "b.txt"
file_a.write_text("Original content A")
file_b.write_text("Original content B")

tracker = IngestionTracker(db_path=db_path)

print("=== First check: both files are new ===")
for f in [file_a, file_b]:
    d = tracker.check(f)
    print(f"{f.name}: should_ingest={d.should_ingest} reason={d.reason}")
    assert d.should_ingest and d.reason == "new"

print("\n=== Mark both as ingested ===")
tracker.mark_ingested(file_a)
tracker.mark_ingested(file_b)

print("\n=== Second check: both unchanged (should be skipped) ===")
for f in [file_a, file_b]:
    d = tracker.check(f)
    print(f"{f.name}: should_ingest={d.should_ingest} reason={d.reason}")
    assert not d.should_ingest and d.reason == "unchanged"

print("\n=== Edit file_a only, re-check both ===")
file_a.write_text("Original content A -- EDITED")
d_a, d_b = tracker.check(file_a), tracker.check(file_b)
print(f"a.txt: should_ingest={d_a.should_ingest} reason={d_a.reason}")
print(f"b.txt: should_ingest={d_b.should_ingest} reason={d_b.reason}")
assert d_a.should_ingest and d_a.reason == "changed"
assert not d_b.should_ingest and d_b.reason == "unchanged"
print("PASS: only the edited file is flagged for re-ingestion")

print("\n=== find_deleted: simulate b.txt removed from the corpus listing ===")
tracker.mark_ingested(file_a)  # re-mark the edited file as ingested
deleted = tracker.find_deleted([file_a])  # only file_a is "still present"
print(f"deleted: {deleted}")
assert canonical_source(file_b) in deleted
print("PASS: b.txt correctly detected as removed from the corpus")

shutil.rmtree(tmp_dir)
print("\nALL TRACKER TESTS PASSED")
