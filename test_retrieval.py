"""End-to-end hybrid retrieval test using REAL OpenAI embeddings (the key
is already validated). Ingests all 4 sample files, then runs two queries
chosen to show why hybrid beats either retriever alone:

  1. An exact-ID query ("P-101") — BM25's strength, embeddings are weak
     on bare codes with little surrounding context.
  2. A paraphrased query with no shared keywords — dense embedding's
     strength, BM25 can't match words that never appear in the source.
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

SAMPLE_DIR = Path(__file__).parent / "sample_data"
tmp_dir = Path(tempfile.mkdtemp())

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

print("=== Ingesting all 4 sample files with real embeddings ===")
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

print("\n=== Query 1: exact policy ID 'P-101' (BM25's strength) ===")
results = retriever.retrieve("P-101", top_k=3)
for r in results:
    print(f"  score={r.score:.4f} source={Path(r.metadata['source']).name} section={r.metadata['section']!r}")
    print(f"    {r.content[:80]}...")
top_sources = [Path(r.metadata["source"]).name for r in results]
assert "policy.csv" in top_sources, "expected the CSV row containing P-101 to surface"
print("PASS: exact-ID query surfaced the right CSV row")

print("\n=== Query 2: paraphrase with no shared keywords (dense's strength) ===")
print("(source text: 'Your laptop will be provisioned within 2 business days')")
results2 = retriever.retrieve("How quickly will I get a new computer when I join?", top_k=3)
for r in results2:
    print(f"  score={r.score:.4f} source={Path(r.metadata['source']).name} section={r.metadata['section']!r}")
    print(f"    {r.content[:80]}...")
top_sources2 = [Path(r.metadata["source"]).name for r in results2]
assert "onboarding.pdf" in top_sources2, "expected the onboarding PDF to surface despite no shared keywords"
print("PASS: paraphrased query with zero shared keywords still surfaced the right chunk")

shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nALL RETRIEVAL TESTS PASSED")
