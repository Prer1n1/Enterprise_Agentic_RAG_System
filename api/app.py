"""FastAPI REST layer — HTTP wrapper around the same ingestion/agent
pipeline main.py's CLI drives.

Route handlers are sync `def`, not `async def`, on purpose: the
underlying LangGraph/embedding/OpenAI calls are all synchronous, and
FastAPI runs sync handlers in a thread pool automatically — rewriting
the whole pipeline as async would be a much bigger change for no real
benefit at this request volume.

Run with: uvicorn api.app:app --reload
"""

from __future__ import annotations

import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings

from agent.graph import build_agent_graph
from api.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from config import EMBEDDING_MODEL, OPENAI_API_KEY
from ingestion.pipeline import SUPPORTED_EXTENSIONS, IngestionResult, ingest_directory
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_vector_store

UPLOAD_DIR = Path("uploads")


class AppState:
    """Holds the pieces that are expensive to build (the BM25 index in
    particular loads every chunk into memory) so they're built ONCE at
    startup and reused across requests — not rebuilt per-request like
    the CLI does, which only ever runs once per invocation."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.chunk_store = ChunkStore()
        self.vector_store = get_vector_store(embeddings=self.embeddings)
        self.retriever = HybridRetriever(self.vector_store, self.chunk_store)
        self.graph = build_agent_graph(self.retriever)

    def rebuild_retriever(self) -> None:
        """Call after ingestion changes the corpus. The BM25 index is an
        in-memory snapshot taken at construction time — new or deleted
        chunks are invisible to keyword search until this runs."""
        self.retriever = HybridRetriever(self.vector_store, self.chunk_store)
        self.graph = build_agent_graph(self.retriever)


# Guards state mutation (ingestion) against concurrent access. A plain
# threading.Lock, not asyncio.Lock: sync route handlers run in FastAPI's
# real OS thread pool, not as coroutines, so this is the correct
# primitive here.
_state_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set — copy .env.example to .env and add your key")
    app.state.rag = AppState()
    yield


app = FastAPI(title="Enterprise Agentic RAG Platform", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """A real check, not a static 'ok' — actually touches both stores so
    a broken DB file or an unreachable Chroma directory shows up here
    instead of surfacing as a confusing 500 on the first real query."""
    state: AppState = app.state.rag
    chunk_store_ok = True
    chunk_count = 0
    try:
        chunk_count = len(state.chunk_store.get_all())
    except Exception:
        chunk_store_ok = False

    vector_store_ok = True
    try:
        state.vector_store.similarity_search("health check", k=1)
    except Exception:
        vector_store_ok = False

    status = "ok" if (chunk_store_ok and vector_store_ok) else "degraded"
    return HealthResponse(
        status=status,
        chunk_store_reachable=chunk_store_ok,
        vector_store_reachable=vector_store_ok,
        chunk_count=chunk_count,
    )


def _ingest_and_persist(directory: Path) -> IngestionResult:
    """Shared by /ingest and /documents/upload — both end in 'ingest this
    directory and persist the result,' so the delete-then-insert
    correctness logic lives in exactly one place instead of two copies
    that could drift apart."""
    with _state_lock:
        state: AppState = app.state.rag
        result = ingest_directory(directory, embeddings=state.embeddings)

        # Same delete-then-insert correctness rule as main.py's CLI: a
        # changed file can produce a different chunk shape than before,
        # so its old chunks must not linger in either store.
        for source in result.ingested_files:
            state.chunk_store.delete_by_source(source)
            delete_by_source(state.vector_store, source)

        if result.chunks:
            state.chunk_store.save_chunks(result.chunks)
            add_chunks(state.vector_store, result.chunks)

        for deleted_source in result.deleted_files:
            state.chunk_store.delete_by_source(deleted_source)
            delete_by_source(state.vector_store, deleted_source)

        if result.ingested_files or result.deleted_files:
            state.rebuild_retriever()

    return result


def _to_ingest_response(result: IngestionResult) -> IngestResponse:
    return IngestResponse(
        ingested_files=len(result.ingested_files),
        skipped_unchanged=len(result.skipped_unchanged),
        deleted_files=len(result.deleted_files),
        chunks_stored=len(result.chunks),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    result = _ingest_and_persist(directory)
    return _to_ingest_response(result)


@app.post("/documents/upload", response_model=IngestResponse)
def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    """Lets a document get into the platform over HTTP, instead of
    requiring filesystem/SSH access to wherever the app happens to be
    running — the same reason a real deployment can't rely on /ingest's
    directory-path approach alone."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-ingesting the whole uploads/ directory, not just the new file, is
    # deliberate: it reuses the SAME incremental-tracker path as directory
    # ingestion (unchanged files there are still skipped via content hash),
    # so there's exactly one ingestion code path to trust, not two.
    result = _ingest_and_persist(UPLOAD_DIR)
    return _to_ingest_response(result)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    with _state_lock:
        graph = app.state.rag.graph  # brief hold just to read a consistent reference

    try:
        result = graph.invoke({"query": request.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}") from e

    return QueryResponse(
        answer=result["answer"],
        categories_queried=result["categories"],
        citations=[
            Citation(source=c["source"], section=c.get("section"), category=c.get("category"))
            for c in result["citations"]
        ],
    )
