# Enterprise Agentic RAG Platform

A multi-agent RAG platform for enterprise knowledge search across PDF, DOCX, HTML, and CSV sources — built with LangChain, LangGraph, and FastAPI-style project structure (API layer in progress).

An LLM router decides which knowledge source(s) (HR / Finance / Security / IT / Legal) are relevant to a question, queries them **in parallel** via a LangGraph workflow, and synthesizes one grounded, cited answer from hybrid (dense + keyword) retrieval results.

See [docs/design-decisions.md](docs/design-decisions.md) for the reasoning behind every architectural choice — what was picked, why, why not the alternatives, and real bugs found while building it.

## Architecture

```
Documents (PDF/DOCX/HTML/CSV)
        |
   [Ingestion]  loaders -> metadata extraction -> hierarchical chunking -> incremental tracker
        |
   [Storage]    Chroma (vectors + metadata filtering)  +  SQLite (chunk text, source of truth)
        |
   [Retrieval]  hybrid: dense (Chroma) + sparse (BM25) fused with Reciprocal Rank Fusion
        |
   [Agent]      LangGraph: plan (route to sources) -> parallel retrieve per source -> synthesize (grounded + cited)
```

This project is classified as **Agentic RAG**: an agent decides which knowledge sources to query rather than always searching everything. See the design log for where it sits relative to Naive/Advanced/Modular/Adaptive-Self-Reflective RAG.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your real OPENAI_API_KEY
```

## Running it

```bash
# Ingest documents into persistent local storage (Chroma + SQLite)
python main.py ingest sample_data

# Ask one question
python main.py ask "How many paid leave days do employees get?"

# Interactive chat loop
python main.py chat
```

Re-running `ingest` is incremental — unchanged files are skipped (content-hash based), and only new/changed files are re-processed.

## Project structure

```
ingestion/    loaders (PDF/DOCX/HTML/CSV) -> common Document schema -> metadata extraction -> chunking -> incremental tracker
storage/      Chroma vector store + SQLite chunk store
retrieval/    BM25 + hybrid retriever (Reciprocal Rank Fusion)
agent/        LangGraph state, router/planner, nodes, graph
main.py       CLI: ingest / ask / chat
docs/         design-decisions.md — the full reasoning log
```

## Testing

Each component has a standalone `test_*.py` script at the project root (no pytest framework yet — these are direct sanity checks with assertions, runnable individually):

```bash
python test_loaders.py
python test_metadata.py
python test_chunking.py
python test_tracker.py
python test_pipeline.py
python test_storage.py
python test_retrieval.py
python test_agent.py
```

## Status

Built so far: Ingestion & Processing, Storage, Retrieval, Agent Orchestration (LangGraph).
Not yet built: Evaluation & Observability (LangSmith tracing, RAGAS, hallucination detection), FastAPI REST layer, Docker packaging.
