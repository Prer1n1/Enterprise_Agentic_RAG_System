"""Incremental ingestion tracker.

Keeps a manifest (SQLite) of every file's content hash and when it was
last ingested. On each ingestion run: unchanged files are skipped
entirely; only new or changed files get (re)chunked and (re)embedded.
Turns "re-ingest the knowledge base" from O(all files) into O(changed
files) — the whole point at enterprise scale.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

from .schema import canonical_source

DEFAULT_DB_PATH = Path(__file__).parent.parent / "storage" / "ingestion_manifest.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_files (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_ingested_at TEXT NOT NULL
)
"""


def file_hash(file_path: Union[str, Path]) -> str:
    """SHA-256 of the raw file bytes. Content-based, not mtime-based:
    touching a file without changing it won't trigger a re-ingest, and
    mtime alone can't tell you that either — only actual byte content can."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IngestDecision:
    file_path: str
    should_ingest: bool
    reason: str  # "new" | "changed" | "unchanged"


class IngestionTracker:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(_SCHEMA)
        return conn

    def check(self, file_path: Union[str, Path]) -> IngestDecision:
        """Decide whether a file needs (re)ingestion. Read-only — does
        not mutate the manifest."""
        file_path_str = canonical_source(file_path)
        current_hash = file_hash(file_path)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT content_hash FROM ingested_files WHERE file_path = ?",
                (file_path_str,),
            ).fetchone()

        if row is None:
            return IngestDecision(file_path_str, True, "new")
        if row[0] != current_hash:
            return IngestDecision(file_path_str, True, "changed")
        return IngestDecision(file_path_str, False, "unchanged")

    def mark_ingested(self, file_path: Union[str, Path]) -> None:
        """Call ONLY after a file has been successfully chunked and
        embedded. Marking earlier would hide a failed ingestion as
        'already done' on the next run — the manifest would lie."""
        file_path_str = canonical_source(file_path)
        current_hash = file_hash(file_path)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO ingested_files (file_path, content_hash, last_ingested_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       content_hash = excluded.content_hash,
                       last_ingested_at = excluded.last_ingested_at""",
                (file_path_str, current_hash, now),
            )
            conn.commit()

    def find_deleted(
        self,
        current_file_paths: List[Union[str, Path]],
        root: Union[str, Path, None] = None,
    ) -> List[str]:
        """Files the manifest still tracks (under `root`, if given) that
        are no longer in the current corpus listing — callers use this to
        purge their chunks from the vector store on the next ingestion run.

        `root` scopes the comparison to one corpus root. Without it, this
        is a global comparison across EVERY file ever tracked — which
        breaks the moment more than one ingestion root shares a tracker
        (verified bug: ingesting uploads/ alone made every sample_data/
        file look "deleted" simply because it wasn't in the uploads/
        listing, and their chunks were wrongly purged from both stores).
        """
        current_set = {canonical_source(p) for p in current_file_paths}
        with closing(self._connect()) as conn:
            tracked = {row[0] for row in conn.execute("SELECT file_path FROM ingested_files")}
        if root is not None:
            root_prefix = canonical_source(root).rstrip("/") + "/"
            tracked = {t for t in tracked if t.startswith(root_prefix)}
        return sorted(tracked - current_set)

    def filter_needs_ingestion(self, file_paths: List[Union[str, Path]]) -> List[IngestDecision]:
        return [self.check(p) for p in file_paths]
