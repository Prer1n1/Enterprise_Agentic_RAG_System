"""Ingestion pipeline — wires the four components together:
tracker (skip unchanged) -> loaders (format -> common schema) ->
metadata extractor (enrich) -> chunking (hierarchical hybrid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from .chunking import Chunk, chunk_documents
from .loaders import load_document
from .metadata_extractor import enrich_all
from .tracker import IngestionTracker

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".csv"}


@dataclass
class IngestionResult:
    chunks: List[Chunk] = field(default_factory=list)
    ingested_files: List[str] = field(default_factory=list)
    skipped_unchanged: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)


def ingest_directory(
    directory: Union[str, Path],
    tracker: IngestionTracker = None,
    embeddings=None,
) -> IngestionResult:
    directory = Path(directory)
    tracker = tracker or IngestionTracker()

    all_files = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    result = IngestionResult(deleted_files=tracker.find_deleted(all_files))

    docs_to_chunk = []
    files_pending_mark = []

    for file_path in all_files:
        decision = tracker.check(file_path)
        if not decision.should_ingest:
            result.skipped_unchanged.append(str(file_path))
            continue

        docs_to_chunk.extend(enrich_all(load_document(file_path)))
        files_pending_mark.append(file_path)

    if docs_to_chunk:
        result.chunks = chunk_documents(docs_to_chunk, embeddings=embeddings)

    # Only mark as ingested AFTER chunking succeeds for all of them —
    # if chunk_documents() raised, nothing here runs and the tracker still
    # reports these files as needing ingestion on the next run.
    for file_path in files_pending_mark:
        tracker.mark_ingested(file_path)
        result.ingested_files.append(str(file_path))

    return result
