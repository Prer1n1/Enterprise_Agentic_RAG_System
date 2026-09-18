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

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
from api.security import require_api_key
from config import (
    API_KEY,
    EMBEDDING_MODEL,
    GOOGLE_DRIVE_CREDENTIALS_PATH,
    GOOGLE_DRIVE_FOLDER_ID,
    OPENAI_API_KEY,
)
from ingestion.pipeline import SUPPORTED_EXTENSIONS, IngestionResult, ingest_directory
from ingestion.tracker import IngestionTracker
from logging_config import configure_logging
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_vector_store

# Called here, not in config.py: config.py is imported by main.py's CLI too,
# and a human watching CLI output wants readable print()s, not JSON log
# lines — structured logging is scoped to the long-running API server only.
configure_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
DRIVE_CACHE_DIR = Path("storage") / "drive_cache"


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
    if not API_KEY:
        raise RuntimeError(
            "API_KEY not set — copy .env.example to .env and set one "
            '(generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    logger.info("startup_begin")
    app.state.rag = AppState()
    logger.info("startup_complete", extra={"chunk_count": len(app.state.rag.chunk_store.get_all())})
    yield
    logger.info("shutdown")


app = FastAPI(title="Enterprise Agentic RAG Platform", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """A real check, not a static 'ok' — actually touches both stores so
    a broken DB file or an unreachable Chroma directory shows up here
    instead of surfacing as a confusing 500 on the first real query.

    Deliberately NOT behind require_api_key: an orchestrator's liveness/
    readiness probe (Docker HEALTHCHECK, a Kubernetes probe, a load
    balancer) needs to reach this without holding a secret — that's the
    standard convention for health endpoints, not a gap in the auth."""
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
    logger.info("ingestion_begin", extra={"directory": str(directory)})
    tracker = IngestionTracker()
    with _state_lock:
        state: AppState = app.state.rag
        result = ingest_directory(directory, tracker=tracker, embeddings=state.embeddings)

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

        # Marked ingested ONLY now — after both stores have actually
        # persisted the new chunks, not right after chunking succeeded.
        # See ingest_directory()'s docstring in ingestion/pipeline.py for
        # the real bug (a tracker that could lie about what's actually
        # persisted) this ordering fixes.
        for file_path in result.pending_mark:
            tracker.mark_ingested(file_path)

        if result.ingested_files or result.deleted_files:
            state.rebuild_retriever()

    logger.info(
        "ingestion_complete",
        extra={
            "directory": str(directory),
            "ingested_files": len(result.ingested_files),
            "skipped_unchanged": len(result.skipped_unchanged),
            "deleted_files": len(result.deleted_files),
            "chunks_stored": len(result.chunks),
        },
    )
    return result


def _to_ingest_response(result: IngestionResult) -> IngestResponse:
    return IngestResponse(
        ingested_files=len(result.ingested_files),
        skipped_unchanged=len(result.skipped_unchanged),
        deleted_files=len(result.deleted_files),
        chunks_stored=len(result.chunks),
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(request: IngestRequest) -> IngestResponse:
    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    result = _ingest_and_persist(directory)
    return _to_ingest_response(result)


@app.post("/documents/upload", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
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
    logger.info("document_uploaded", extra={"filename": file.filename})

    # Re-ingesting the whole uploads/ directory, not just the new file, is
    # deliberate: it reuses the SAME incremental-tracker path as directory
    # ingestion (unchanged files there are still skipped via content hash),
    # so there's exactly one ingestion code path to trust, not two.
    result = _ingest_and_persist(UPLOAD_DIR)
    return _to_ingest_response(result)


@app.post("/ingest/drive", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest_drive() -> IngestResponse:
    """Syncs the configured Google Drive folder into a local cache, then
    reuses the identical ingest_directory() path as everything else —
    the connector's only job is making Drive content look like local
    files (see ingestion/connectors/google_drive.py)."""
    if not GOOGLE_DRIVE_CREDENTIALS_PATH or not GOOGLE_DRIVE_FOLDER_ID:
        raise HTTPException(
            status_code=400,
            detail="GOOGLE_DRIVE_CREDENTIALS_PATH and GOOGLE_DRIVE_FOLDER_ID must be set in .env",
        )

    from ingestion.connectors.google_drive import GoogleDriveConnector

    logger.info("drive_sync_begin", extra={"folder_id": GOOGLE_DRIVE_FOLDER_ID})
    connector = GoogleDriveConnector(GOOGLE_DRIVE_CREDENTIALS_PATH, GOOGLE_DRIVE_FOLDER_ID)
    connector.sync_to_local(DRIVE_CACHE_DIR)

    result = _ingest_and_persist(DRIVE_CACHE_DIR)
    return _to_ingest_response(result)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    with _state_lock:
        graph = app.state.rag.graph  # brief hold just to read a consistent reference

    # question_preview, not the full question: keeps log lines short and
    # scannable without truncating anything that actually matters for
    # debugging (the routed categories + citation sources below cover that).
    logger.info("query_received", extra={"question_preview": request.question[:80]})

    try:
        result = graph.invoke({"query": request.question})
    except Exception:
        # logger.exception (not logger.error) captures the full traceback in
        # the structured log — the client only ever sees a generic 500, but
        # whoever's watching these logs gets the real failure, not just "it
        # broke."
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail="Agent execution failed") from None

    logger.info(
        "query_answered",
        extra={"categories_queried": result["categories"], "citation_count": len(result["citations"])},
    )
    return QueryResponse(
        answer=result["answer"],
        categories_queried=result["categories"],
        citations=[
            Citation(source=c["source"], section=c.get("section"), category=c.get("category"))
            for c in result["citations"]
        ],
    )
