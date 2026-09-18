"""Chunking — hierarchical hybrid strategy.

1. Structure-aware boundaries already exist: loaders already split content
   by page/section/row, so that work is done before this module runs.
2. Prose Documents get SEMANTIC chunking: split into sentences, embed each
   one, and cut where consecutive sentences suddenly diverge in meaning —
   not at a fixed token count.
3. Tabular Documents (CSV rows, DOCX/HTML tables) get BATCHED, not
   semantically chunked — there's no "meaning similarity" between rows to
   detect, so rows are simply grouped up to a size cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings

from retry_utils import retry_openai_call

from .schema import Document, DocumentMetadata

MAX_TABLE_CHUNK_CHARS = 1500
SEMANTIC_BREAKPOINT_PERCENTILE = 90  # cut at the sharpest 10% of meaning "jumps"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    content: str
    metadata: DocumentMetadata
    chunk_index: int


def _is_tabular(document: Document) -> bool:
    return document.metadata.doc_type == "csv" or document.metadata.extra.get("is_table", False)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    similarity = np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return 1 - similarity


@retry_openai_call
def _embed_documents(embeddings, sentences: List[str]) -> List[List[float]]:
    """A flaky OpenAI embedding call mid-ingestion shouldn't abort an
    entire (potentially large) batch of files — retried a few times with
    backoff before actually failing. See retry_utils.py."""
    return embeddings.embed_documents(sentences)


def semantic_split(
    text: str,
    embeddings,
    breakpoint_percentile: int = SEMANTIC_BREAKPOINT_PERCENTILE,
) -> List[str]:
    """The actual "semantic" part: embed every sentence, measure cosine
    distance between each consecutive pair, and cut wherever that distance
    is in the top (100 - breakpoint_percentile)% — i.e. an unusually large
    jump in meaning, which signals a real topic shift rather than just a
    new sentence."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    vectors = _embed_documents(embeddings, sentences)
    distances = [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = np.percentile(distances, breakpoint_percentile)

    pieces: List[str] = []
    current = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            pieces.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    pieces.append(" ".join(current))
    return pieces


def _chunk_prose(documents: List[Document], embeddings) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        for piece in semantic_split(doc.content, embeddings):
            chunks.append(Chunk(content=piece, metadata=doc.metadata, chunk_index=len(chunks)))
    return chunks


def _combine_batch(docs: List[Document]) -> Chunk:
    combined = "\n---\n".join(d.content for d in docs)
    first, last = docs[0], docs[-1]
    section = (
        first.metadata.section
        if len(docs) == 1
        else f"{first.metadata.section} .. {last.metadata.section}"
    )
    metadata = DocumentMetadata(
        source=first.metadata.source,
        doc_type=first.metadata.doc_type,
        title=first.metadata.title,
        author=first.metadata.author,
        created_date=first.metadata.created_date,
        section=section,
        extra={**first.metadata.extra, "batched_rows": len(docs)},
    )
    return Chunk(content=combined, metadata=metadata, chunk_index=0)


def _source_category_key(document: Document):
    return document.metadata.source, document.metadata.extra.get("category", "")


def _chunk_tabular(documents: List[Document], max_chars: int = MAX_TABLE_CHUNK_CHARS) -> List[Chunk]:
    """Groups consecutive rows/tables from the same source up to a size
    cap — never one chunk per row (too tiny to be useful retrieval
    context) and never the whole table in one chunk (can blow past
    context-window limits on large tables).

    Grouped by (source, category), not just source — verified bug: rows
    with DIFFERENT categories batched together got the whole chunk tagged
    with only the first row's category, making the other rows' content
    invisible to category-filtered retrieval (an IT-tagged CSV row about
    password rotation was hidden inside a chunk labeled "Finance").
    """
    chunks: List[Chunk] = []
    sorted_docs = sorted(documents, key=_source_category_key)  # stable: preserves row order within a group
    for _, group_iter in groupby(sorted_docs, key=_source_category_key):
        group = list(group_iter)
        buffer: List[Document] = []
        buffer_len = 0
        for doc in group:
            if buffer and buffer_len + len(doc.content) > max_chars:
                chunks.append(_combine_batch(buffer))
                buffer, buffer_len = [], 0
            buffer.append(doc)
            buffer_len += len(doc.content)
        if buffer:
            chunks.append(_combine_batch(buffer))

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def chunk_documents(
    documents: List[Document],
    embeddings: Optional[object] = None,
    max_table_chunk_chars: int = MAX_TABLE_CHUNK_CHARS,
) -> List[Chunk]:
    embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
    tabular = [d for d in documents if _is_tabular(d)]
    prose = [d for d in documents if not _is_tabular(d)]
    combined = _chunk_tabular(tabular, max_table_chunk_chars) + _chunk_prose(prose, embeddings)

    # Each helper indexes its own chunks from 0 independently, so after
    # concatenation chunk_index is no longer unique across the full list —
    # reindex so it reflects final position (compute_chunk_id uses it, and
    # a duplicate index paired with a duplicate section string would
    # collide two different chunks onto the same chunk_id).
    for i, chunk in enumerate(combined):
        chunk.chunk_index = i
    return combined
