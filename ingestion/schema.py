"""Common document schema every format-specific loader normalizes into.

Why this exists: PDF/DOCX/HTML/CSV each have wildly different native
structures (pages, paragraphs+headings, DOM tags, rows). Every downstream
component (chunking, metadata, retrieval) should only ever see ONE shape,
not four. This is what makes the pipeline format-agnostic past this point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


def canonical_source(file_path: Union[str, Path]) -> str:
    """The identity string used for a source file EVERYWHERE it matters
    for correctness: metadata.source, the incremental tracker's keys, and
    chunk_id derivation. Always forward-slash (POSIX), regardless of OS.

    Why this exists: verified bug — ingesting the same file from Windows
    (str(Path) -> "sample_data\\x.csv") and from inside a Linux Docker
    container (str(Path) -> "sample_data/x.csv") produced two different
    identity strings for the identical file. The incremental tracker
    treated the container run as "4 new files" AND "4 deleted files"
    simultaneously, silently defeating the whole point of incremental
    ingestion across the exact environments this project is meant to run
    in (a Windows dev machine building an image that runs on Linux).
    Only used for IDENTITY strings — actually opening a file still uses
    the native str(file_path), which must stay OS-correct.
    """
    return Path(file_path).as_posix()


@dataclass
class DocumentMetadata:
    source: str                    # file path the content came from
    doc_type: str                  # "pdf" | "docx" | "html" | "csv"
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    section: Optional[str] = None  # heading/section name, or "Row N" for CSV
    page_number: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """One loaded unit of content (a PDF page, a DOCX section, an HTML
    section, a CSV row) before chunking splits it further."""

    content: str
    metadata: DocumentMetadata

    def content_hash(self) -> str:
        """SHA-256 of the content — used later by the incremental
        ingestion tracker to detect whether a document actually changed."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
