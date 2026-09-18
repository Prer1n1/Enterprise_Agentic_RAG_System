"""Request/response models for the REST API — the HTTP-facing contract,
kept separate from the internal dataclasses (Chunk, RetrievedChunk, etc.)
so internal refactors don't silently change the public API shape.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source: str
    section: Optional[str] = None
    category: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    categories_queried: List[str]
    citations: List[Citation]
    access_restricted: bool = Field(
        default=False,
        description="True if the caller's API key scope narrowed which knowledge-source categories could be searched for this query.",
    )


class IngestRequest(BaseModel):
    directory: str = "sample_data"


class IngestResponse(BaseModel):
    ingested_files: int
    skipped_unchanged: int
    deleted_files: int
    chunks_stored: int
    blocked_files: List[str] = Field(
        default_factory=list,
        description="Files whose content was flagged as a suspected prompt injection attempt and blocked from ingestion.",
    )
    pii_check_failed: List[str] = Field(
        default_factory=list,
        description="Files blocked because the PII detector itself failed (fail-closed) — not known to be malicious, just unverified.",
    )


class HealthResponse(BaseModel):
    status: str
    chunk_store_reachable: bool
    vector_store_reachable: bool
    chunk_count: int
