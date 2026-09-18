"""Vector store wrapper — Chroma, persisted locally to disk.

Why Chroma over FAISS: FAISS is a pure similarity index with no native
metadata storage, so filtering by category/doc_type/source before or
after search would need to be hand-rolled separately. Chroma stores
metadata alongside each vector and supports filtering directly in the
query — a real requirement for enterprise search ("only search Security
department docs"), not a nice-to-have.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument

from ingestion.chunking import Chunk
from retry_utils import retry_openai_call
from storage.chunk_store import compute_chunk_id

DEFAULT_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "enterprise_rag"


def get_vector_store(embeddings, persist_directory: Union[str, Path] = DEFAULT_PERSIST_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def _to_langchain_document(chunk: Chunk) -> LangchainDocument:
    m = chunk.metadata
    # Chroma metadata values must be str/int/float/bool — flatten or drop
    # anything nested (e.g. extra["columns"] is a list).
    metadata = {
        "chunk_id": compute_chunk_id(chunk),
        "source": m.source,
        "doc_type": m.doc_type,
        "title": m.title or "",
        "section": m.section or "",
        "page_number": m.page_number or 0,
        "doc_id": m.extra.get("doc_id", ""),
        "category": m.extra.get("category", ""),
        "language": m.extra.get("language", ""),
    }
    return LangchainDocument(page_content=chunk.content, metadata=metadata)


@retry_openai_call
def _add_documents(vector_store: Chroma, docs: List[LangchainDocument], ids: List[str]) -> None:
    """add_documents() embeds every doc via OpenAI under the hood — a
    flaky call here is exactly as disruptive mid-ingestion as the one in
    chunking.py's semantic split. See retry_utils.py."""
    vector_store.add_documents(documents=docs, ids=ids)


def add_chunks(vector_store: Chroma, chunks: List[Chunk]) -> List[str]:
    if not chunks:
        return []
    # Same chunk_id function as ChunkStore, so a chunk has ONE identity
    # across both stores — required for delete_by_source to stay in sync.
    ids = [compute_chunk_id(c) for c in chunks]
    docs = [_to_langchain_document(c) for c in chunks]
    _add_documents(vector_store, docs, ids)
    return ids


def delete_by_source(vector_store: Chroma, source: str) -> None:
    vector_store.delete(where={"source": source})


@retry_openai_call
def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 5,
    filter: Optional[dict] = None,
) -> List[LangchainDocument]:
    """Dense retrieval embeds the query text via OpenAI before searching
    Chroma — retried the same way as every other OpenAI-dependent call in
    this project, since a query-time failure here directly fails a user's
    /query request. See retry_utils.py."""
    return vector_store.similarity_search(query, k=k, filter=filter)
