"""Chunk store — the source-of-truth record for every chunk's text and
metadata (SQLite). The vector store (Chroma) holds embeddings for
similarity search, but this table is what lets us: rebuild the BM25
keyword index later, reconstruct a document's chunks in order for
citation context, and delete a source's old chunks in one place when it
changes or is removed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

from ingestion.chunking import Chunk

DEFAULT_DB_PATH = Path(__file__).parent / "chunk_store.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    doc_id TEXT,
    doc_type TEXT,
    section TEXT,
    page_number INTEGER,
    category TEXT,
    language TEXT,
    chunk_index INTEGER,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
"""


def compute_chunk_id(chunk: Chunk) -> str:
    """Deterministic, not random — re-ingesting the same source/section/
    index combo produces the same ID, so an UPSERT naturally replaces the
    old record instead of creating a duplicate. Also lets the vector store
    use the SAME id for the same chunk, keeping both stores in sync."""
    raw = f"{chunk.metadata.source}::{chunk.metadata.section}::{chunk.chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ChunkStore:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(_SCHEMA)
        return conn

    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Upserts chunks, returns their chunk_ids."""
        now = datetime.now(timezone.utc).isoformat()
        chunk_ids = []
        with closing(self._connect()) as conn:
            for chunk in chunks:
                chunk_id = compute_chunk_id(chunk)
                chunk_ids.append(chunk_id)
                m = chunk.metadata
                conn.execute(
                    """INSERT INTO chunks (chunk_id, source, doc_id, doc_type, section,
                                            page_number, category, language, chunk_index,
                                            content, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                           content = excluded.content,
                           created_at = excluded.created_at""",
                    (
                        chunk_id, m.source, m.extra.get("doc_id"), m.doc_type, m.section,
                        m.page_number, m.extra.get("category"), m.extra.get("language"),
                        chunk.chunk_index, chunk.content, now,
                    ),
                )
            conn.commit()
        return chunk_ids

    def delete_by_source(self, source: str) -> int:
        """Removes every chunk for one source file — called before
        re-inserting a changed file's new chunks, or when the ingestion
        tracker reports the file no longer exists."""
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            conn.commit()
            return cur.rowcount

    def get_all(self) -> List[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM chunks").fetchall()
            return [dict(r) for r in rows]

    def get_by_source(self, source: str) -> List[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source = ? ORDER BY chunk_index", (source,)
            ).fetchall()
            return [dict(r) for r in rows]
