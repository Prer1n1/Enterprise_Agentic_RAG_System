"""End-to-end test of the LangGraph agent workflow with REAL OpenAI calls
(embeddings + chat). Ingests all sample docs, builds the graph, and runs
two queries chosen to show the router actually making different decisions:

  1. A focused HR question -> should route to HR (maybe + General).
  2. A cross-domain question -> should route to multiple categories.
"""

import shutil
import tempfile
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL
from ingestion.loaders import load_document
from ingestion.metadata_extractor import enrich_all
from ingestion.chunking import chunk_documents
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, get_vector_store
from retrieval.hybrid_retriever import HybridRetriever
from agent.planner import plan_categories
from agent.graph import build_agent_graph

SAMPLE_DIR = Path(__file__).parent / "sample_data"
tmp_dir = Path(tempfile.mkdtemp())

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print("=== Ingesting all 4 sample files ===")
all_docs = []
for fname in ["onboarding.pdf", "security_policy.docx", "handbook.html", "policy.csv"]:
    all_docs.extend(enrich_all(load_document(SAMPLE_DIR / fname)))

chunks = chunk_documents(all_docs, embeddings=embeddings)
print(f"{len(all_docs)} Documents -> {len(chunks)} chunks")

chunk_store = ChunkStore(db_path=tmp_dir / "chunks.db")
chunk_store.save_chunks(chunks)
vector_store = get_vector_store(embeddings=embeddings, persist_directory=tmp_dir / "chroma_db")
add_chunks(vector_store, chunks)

retriever = HybridRetriever(vector_store, chunk_store)
graph = build_agent_graph(retriever)

print("\n=== Router test: focused HR question ===")
categories, off_topic = plan_categories("How many paid leave days do employees get per year?")
print(f"Routed to: {categories} (off_topic={off_topic})")
assert "HR" in categories, f"expected HR in routing decision, got {categories}"
assert off_topic is False
print("PASS: focused question routed to HR")

print("\n=== Router test: genuinely off-topic question ===")
categories, off_topic = plan_categories("Write me a short poem about the ocean.")
print(f"Routed to: {categories} (off_topic={off_topic})")
assert off_topic is True, "an unrelated creative-writing request should be recognized as off-topic"
assert categories == []
print("PASS: off-topic question correctly short-circuited instead of searching every category")

print("\n=== Full graph run: focused HR question ===")
result = graph.invoke({"query": "How many paid leave days do employees get per year?"})
print(f"Categories queried: {result['categories']}")
print(f"Chunks retrieved (deduped later in synth): {len(result['retrieved_chunks'])}")
print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
assert "20" in result["answer"], "expected the answer to mention the actual number (20 days) from the source"
print("PASS: answer is grounded in the actual retrieved number")

print("\n=== Full graph run: cross-domain question ===")
result2 = graph.invoke({"query": "What security and expense rules do I need to follow as a new employee?"})
print(f"Categories queried: {result2['categories']}")
print(f"Answer: {result2['answer']}")
print(f"Citations: {result2['citations']}")
assert len(result2["categories"]) >= 2, f"expected a cross-domain question to route to 2+ categories, got {result2['categories']}"
print("PASS: cross-domain question routed to multiple knowledge sources")

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nALL AGENT TESTS PASSED")
