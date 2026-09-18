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
from .schema import canonical_source
from .tracker import IngestionTracker

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".csv"}


@dataclass
class IngestionResult:
    chunks: List[Chunk] = field(default_factory=list)
    ingested_files: List[str] = field(default_factory=list)
    skipped_unchanged: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    # Real Path objects (tracker.mark_ingested() needs to reopen the file to
    # hash it), NOT yet marked in the tracker — see the note on
    # ingest_directory() below for why marking is the CALLER's job now.
    pending_mark: List[Path] = field(default_factory=list)


def ingest_directory(
    directory: Union[str, Path],
    tracker: IngestionTracker = None,
    embeddings=None,
    classifier_llm=None,
    use_llm_classifier: bool = True,
) -> IngestionResult:
    directory = Path(directory)
    tracker = tracker or IngestionTracker()

    # rglob, not iterdir: real document corpora are organized into
    # subfolders (documents/HR/, documents/Finance/, ...) — iterdir() only
    # lists a folder's immediate children and silently misses everything
    # nested deeper, with no error to indicate anything was skipped.
    all_files = [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    # root=directory scopes deletion-detection to just this corpus root —
    # required now that /ingest and /documents/upload can point at
    # different directories (sample_data/, uploads/, ...) sharing one
    # tracker. Without it, ingesting one root wrongly looks like every
    # file in every OTHER root was deleted.
    result = IngestionResult(deleted_files=tracker.find_deleted(all_files, root=directory))

    docs_to_chunk = []
    files_pending_mark = []

    for file_path in all_files:
        decision = tracker.check(file_path)
        if not decision.should_ingest:
            result.skipped_unchanged.append(canonical_source(file_path))
            continue

        docs_to_chunk.extend(
            enrich_all(load_document(file_path), llm=classifier_llm, use_llm=use_llm_classifier)
        )
        files_pending_mark.append(file_path)

    if docs_to_chunk:
        result.chunks = chunk_documents(docs_to_chunk, embeddings=embeddings)

    # NOT marked here anymore — see the SEVERE bug this fixed in
    # docs/design-decisions.md ("Reliability / Data Integrity"). Marking
    # a file "ingested" the moment chunking succeeds, before its chunks
    # are actually saved to chunk_store/vector_store, meant a crash or a
    # failed save between here and persistence would leave the tracker
    # believing a file was ingested when its chunks were never actually
    # stored anywhere — silently and permanently, since the next run's
    # tracker.check() would report it "unchanged" and skip it forever.
    #
    # The caller (main.py / api/app.py) now calls tracker.mark_ingested()
    # itself, ONLY after chunk_store.save_chunks() and vector_store's
    # add_chunks() have BOTH actually succeeded — that's the real
    # definition of "ingested," not "chunked."
    for file_path in files_pending_mark:
        result.pending_mark.append(file_path)
        # MUST match the canonical_source() used in Document.metadata.source
        # (set by the loaders) — main.py/api.py call delete_by_source() on
        # these entries before persisting new chunks, and that only finds
        # the right rows if the identity string matches exactly.
        result.ingested_files.append(canonical_source(file_path))

    return result
