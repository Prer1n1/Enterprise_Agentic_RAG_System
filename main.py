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
from config import EMBEDDING_MODEL, OPENAI_API_KEY
from ingestion.pipeline import ingest_directory
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_vector_store


def _require_api_key() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Copy .env.example to .env and add your real key.")
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    _require_api_key()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)

    directory = Path(args.directory)
    print(f"Ingesting: {directory}")
    result = ingest_directory(directory, embeddings=embeddings)

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

    print(f"Ingested (new/changed): {len(result.ingested_files)} file(s)")
    print(f"Skipped (unchanged):    {len(result.skipped_unchanged)} file(s)")
    print(f"Purged (deleted):       {len(result.deleted_files)} file(s)")
    print(f"Chunks stored:          {len(result.chunks)}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Agentic RAG Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into persistent storage")
    ingest_parser.add_argument("directory", nargs="?", default="sample_data")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A loop")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
