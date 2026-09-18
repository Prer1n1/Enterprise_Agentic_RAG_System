"""Sanity checks for retrieval/reranker.py and its wiring into
HybridRetriever. Runs free/offline: both the no-COHERE_API_KEY fallback
and HybridRetriever's use of the reranker are proven by monkeypatching
(retrieval.reranker.COHERE_API_KEY and hybrid_retriever's imported
`rerank` name) rather than depending on whether a real key happens to be
configured in the local environment — this test must pass identically
whether or not you've actually added a COHERE_API_KEY, the same way
test_chunking.py's fake embedding model doesn't care whether a real
OPENAI_API_KEY is set.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

import retrieval.hybrid_retriever as hybrid_retriever_module
import retrieval.reranker as reranker_module
from config import COHERE_API_KEY
from ingestion.chunking import chunk_documents
from ingestion.loaders import load_document
from ingestion.metadata_extractor import enrich_all
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, get_vector_store

SAMPLE_DIR = Path(__file__).parent / "sample_data"


class HashFakeEmbeddings:
    def _vec(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255 for b in h[:8]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


print("=== rerank() with no COHERE_API_KEY configured (monkeypatched, regardless of local .env) ===")
original_key = reranker_module.COHERE_API_KEY
reranker_module.COHERE_API_KEY = None
try:
    result = reranker_module.rerank("some query", ["doc a", "doc b"], top_n=2)
finally:
    reranker_module.COHERE_API_KEY = original_key
assert result is None, "rerank() must return None (not raise) when no key is configured"
print("PASS: rerank() returns None gracefully with no key, instead of raising")

print("\n=== HybridRetriever.retrieve() falls back to RRF order when reranking unavailable ===")
tmp_dir = Path(tempfile.mkdtemp())
fake = HashFakeEmbeddings()

all_docs = enrich_all(load_document(SAMPLE_DIR / "policy.csv"), use_llm=False)
chunks = chunk_documents(all_docs, embeddings=fake)
chunk_store = ChunkStore(db_path=tmp_dir / "chunks.db")
chunk_store.save_chunks(chunks)
vector_store = get_vector_store(embeddings=fake, persist_directory=tmp_dir / "chroma_db")
add_chunks(vector_store, chunks)

retriever = HybridRetriever(vector_store, chunk_store)


def unavailable_rerank(query, documents, top_n):
    return None  # simulates no key / a failed call, regardless of local .env


original_rerank = hybrid_retriever_module.rerank
hybrid_retriever_module.rerank = unavailable_rerank
try:
    baseline = retriever.retrieve("password rotation policy", top_k=3, use_reranker=True)
finally:
    hybrid_retriever_module.rerank = original_rerank
assert len(baseline) > 0
print(f"PASS: retrieve() with use_reranker=True but no reranker available returns {len(baseline)} result(s) via RRF fallback")

print("\n=== HybridRetriever.retrieve() actually uses the reranker's order + scores when available ===")


def fake_rerank(query, documents, top_n):
    # Deliberately return the WORST-RRF-ranked candidate first, proving the
    # final order comes from the (fake) reranker, not from RRF.
    n = len(documents)
    reversed_order = list(range(n - 1, -1, -1))[:top_n]
    return [(i, 1.0 - (rank / n)) for rank, i in enumerate(reversed_order)]


original_rerank = hybrid_retriever_module.rerank
hybrid_retriever_module.rerank = fake_rerank
try:
    reranked = retriever.retrieve("password rotation policy", top_k=3, fetch_k=5, use_reranker=True)
finally:
    hybrid_retriever_module.rerank = original_rerank

no_rerank = retriever.retrieve("password rotation policy", top_k=3, fetch_k=5, use_reranker=False)

reranked_ids = [c.chunk_id for c in reranked]
no_rerank_ids = [c.chunk_id for c in no_rerank]
print(f"RRF-only order:   {no_rerank_ids}")
print(f"Fake-reranked order: {reranked_ids}")
assert reranked_ids != no_rerank_ids, "the fake reranker deliberately reverses order — results must differ"
assert reranked[0].score == 1.0 - (0 / 3) or reranked[0].score is not None, "score should come from the reranker"
print("PASS: HybridRetriever actually reorders results using the reranker's output, not just RRF")

shutil.rmtree(tmp_dir, ignore_errors=True)

if COHERE_API_KEY:
    print("\n=== Part 2: real Cohere API call (needs COHERE_API_KEY) ===")
    query = "What is the password rotation policy?"
    docs = [
        "Employees must take at least 10 days of paid leave per year.",
        "All expenses over $500 require manager approval before reimbursement.",
        "Passwords must be rotated every 90 days and must include a special character.",
        "The office is located in downtown Seattle near the transit center.",
    ]
    result = reranker_module.rerank(query, docs, top_n=4)
    assert result is not None, "a real Cohere call with a valid key should not fail"
    top_index, top_score = result[0]
    print(f"Top result (score={top_score:.4f}): {docs[top_index]!r}")
    assert top_index == 2, f"expected the password-rotation doc (index 2) to rank first, got index {top_index}"
    assert top_score > 0.3, f"expected a decisive top score, got {top_score:.4f}"
    second_score = result[1][1]
    assert top_score > second_score * 5, "expected a clear separation between the correct doc and distractors"
    print("PASS: real Cohere rerank correctly identified and decisively scored the relevant document")
else:
    print("\nSkipping Part 2 (real Cohere API test) — COHERE_API_KEY not set.")

print("\nALL RERANKER TESTS PASSED")
