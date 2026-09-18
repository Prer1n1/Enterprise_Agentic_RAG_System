"""Command-line entry point — this is what actually running the platform
looks like, as opposed to the test_*.py scripts which prove individual
pieces work.

    python main.py ingest [directory]   # ingest documents into PERSISTENT storage
    python main.py ask "question"       # ask one question, print the grounded answer
    python main.py chat                 # interactive Q&A loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from agent.graph import build_agent_graph
from config import (
    EMBEDDING_MODEL,
    GOOGLE_DRIVE_CREDENTIALS_PATH,
    GOOGLE_DRIVE_FOLDER_ID,
    OPENAI_API_KEY,
)
from ingestion.pipeline import IngestionResult, ingest_directory
from ingestion.tracker import IngestionTracker
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_vector_store

DRIVE_CACHE_DIR = Path("storage") / "drive_cache"


def _require_api_key() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Copy .env.example to .env and add your real key.")
        sys.exit(1)


def _ingest_and_persist(directory: Path, chunk_store: ChunkStore, vector_store, embeddings) -> IngestionResult:
    """Shared by cmd_ingest and cmd_ingest_drive — both end in 'ingest
    this directory and persist the result,' so the delete-then-insert
    correctness logic (and the API's identical version of it) all trace
    back to the same reasoning, not three copies that could drift apart."""
    tracker = IngestionTracker()
    result = ingest_directory(directory, tracker=tracker, embeddings=embeddings)

    # A re-ingested (changed) file may now produce a different number/shape
    # of chunks than before — delete its OLD chunks first so nothing stale
    # is left orphaned in either store, then insert the fresh ones.
    for source in result.ingested_files:
        chunk_store.delete_by_source(source)
        delete_by_source(vector_store, source)

    if result.chunks:
        chunk_store.save_chunks(result.chunks)
        add_chunks(vector_store, result.chunks)

    for deleted_source in result.deleted_files:
        chunk_store.delete_by_source(deleted_source)
        delete_by_source(vector_store, deleted_source)

    # Marked ingested ONLY now — after chunk_store AND vector_store have
    # both actually persisted the new chunks, not right after chunking
    # succeeded. See ingest_directory()'s docstring in ingestion/pipeline.py
    # for the real bug this ordering fixes.
    for file_path in result.pending_mark:
        tracker.mark_ingested(file_path)

    return result


def _print_ingest_result(result: IngestionResult) -> None:
    print(f"Ingested (new/changed): {len(result.ingested_files)} file(s)")
    print(f"Skipped (unchanged):    {len(result.skipped_unchanged)} file(s)")
    print(f"Purged (deleted):       {len(result.deleted_files)} file(s)")
    print(f"Chunks stored:          {len(result.chunks)}")
    if result.blocked_files:
        print(f"BLOCKED (suspected prompt injection): {len(result.blocked_files)} file(s)")
        for source in result.blocked_files:
            print(f"  - {source}")
    if result.pii_check_failed:
        print(f"BLOCKED (PII check failed, fail-closed): {len(result.pii_check_failed)} file(s)")
        for source in result.pii_check_failed:
            print(f"  - {source}")


def cmd_ingest(args: argparse.Namespace) -> None:
    _require_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)

    directory = Path(args.directory)
    print(f"Ingesting: {directory}")
    result = _ingest_and_persist(directory, chunk_store, vector_store, embeddings)
    _print_ingest_result(result)


def cmd_ingest_drive(args: argparse.Namespace) -> None:
    _require_api_key()
    if not GOOGLE_DRIVE_CREDENTIALS_PATH or not GOOGLE_DRIVE_FOLDER_ID:
        print("ERROR: GOOGLE_DRIVE_CREDENTIALS_PATH and GOOGLE_DRIVE_FOLDER_ID must be set in .env.")
        sys.exit(1)

    from ingestion.connectors.google_drive import GoogleDriveConnector

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)

    print(f"Syncing Google Drive folder {GOOGLE_DRIVE_FOLDER_ID} -> {DRIVE_CACHE_DIR}")
    connector = GoogleDriveConnector(GOOGLE_DRIVE_CREDENTIALS_PATH, GOOGLE_DRIVE_FOLDER_ID)
    connector.sync_to_local(DRIVE_CACHE_DIR)

    print(f"Ingesting: {DRIVE_CACHE_DIR}")
    result = _ingest_and_persist(DRIVE_CACHE_DIR, chunk_store, vector_store, embeddings)
    _print_ingest_result(result)


def _build_agent():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)
    retriever = HybridRetriever(vector_store, chunk_store)
    return build_agent_graph(retriever)


def _print_answer(result: dict) -> None:
    print(f"\nCategories queried: {result['categories']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\nCitations:")
    for c in result["citations"]:
        print(f"  - {Path(c['source']).name} ({c['section'] or 'n/a'}) [{c['category']}]")


def cmd_ask(args: argparse.Namespace) -> None:
    _require_api_key()
    graph = _build_agent()
    result = graph.invoke({"query": args.question})
    _print_answer(result)


def cmd_chat(args: argparse.Namespace) -> None:
    _require_api_key()
    graph = _build_agent()
    print("Enterprise Agentic RAG — type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = graph.invoke({"query": question})
        _print_answer(result)


def cmd_evaluate(args: argparse.Namespace) -> None:
    _require_api_key()
    from evaluation.ragas_eval import print_report, run_evaluation

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)
    retriever = HybridRetriever(vector_store, chunk_store)

    print("Running evaluation (RAGAS: faithfulness, relevancy, context precision)...")
    results = run_evaluation(retriever)
    print_report(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Agentic RAG Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into persistent storage")
    ingest_parser.add_argument("directory", nargs="?", default="sample_data")
    ingest_parser.set_defaults(func=cmd_ingest)

    ingest_drive_parser = subparsers.add_parser("ingest-drive", help="Sync + ingest a Google Drive folder")
    ingest_drive_parser.set_defaults(func=cmd_ingest_drive)

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A loop")
    chat_parser.set_defaults(func=cmd_chat)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run RAGAS evaluation over the curated eval set")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
