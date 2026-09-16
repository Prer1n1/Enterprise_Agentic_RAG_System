"""Sanity checks for chunking.py.

Two parts that need no API key at all (fast, free, deterministic):
  1. semantic_split() tested against a FAKE embedding model with known
     vectors, to prove the boundary-detection math is correct.
  2. _chunk_tabular() tested against real loaded CSV/table Documents.

A third part exercises the full pipeline with real OpenAI embeddings —
only runs if OPENAI_API_KEY is set in .env.
"""

from pathlib import Path

from config import OPENAI_API_KEY
from ingestion.chunking import _chunk_tabular, chunk_documents, semantic_split
from ingestion.loaders import load_document

SAMPLE_DIR = Path(__file__).parent / "sample_data"


class FakeEmbeddings:
    """Returns a fixed vector per exact sentence text — lets us construct
    a controlled test instead of depending on a real model's opinion."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[t] for t in texts]


def test_semantic_split():
    print("\n=== semantic_split() with fake embeddings ===")
    cat_1 = "The cat sat on the mat."
    cat_2 = "The cat is a small mammal."
    stock_1 = "Stock prices rose sharply today."
    stock_2 = "The market rally continued into the afternoon."

    text = " ".join([cat_1, cat_2, stock_1, stock_2])
    fake = FakeEmbeddings({
        cat_1: [1.0, 0.0, 0.0],
        cat_2: [0.95, 0.05, 0.0],   # close to cat_1 -> should merge
        stock_1: [0.0, 1.0, 0.0],   # far from cat_2 -> should split here
        stock_2: [0.0, 0.95, 0.05], # close to stock_1 -> should merge
    })

    pieces = semantic_split(text, fake, breakpoint_percentile=50)
    print(f"Input: 2 cat sentences + 2 stock sentences (4 total)")
    print(f"Output: {len(pieces)} piece(s)")
    for i, p in enumerate(pieces, 1):
        print(f"  [{i}] {p}")

    assert len(pieces) == 2, f"expected a split between topics, got {len(pieces)} piece(s)"
    assert "cat" in pieces[0] and "cat" in pieces[0]
    assert "market" in pieces[1] or "Stock" in pieces[1]
    print("PASS: split landed exactly between the cat sentences and the stock sentences")


def test_chunk_tabular():
    print("\n=== _chunk_tabular() on real CSV rows ===")
    csv_docs = load_document(SAMPLE_DIR / "policy.csv")

    # each row is ~90-120 chars; cap=300 fits 2 rows together but not all 3 ->
    # proves rows actually get GROUPED, not just isolated at a too-small cap
    chunks = _chunk_tabular(csv_docs, max_chars=300)
    print(f"3 CSV rows, max_chars=300 -> {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"  section={c.metadata.section!r} batched_rows={c.metadata.extra['batched_rows']}")
        print(f"    {c.content[:80]}...")

    assert len(chunks) == 2, f"expected rows 1+2 grouped and row 3 alone, got {len(chunks)} chunk(s)"
    assert chunks[0].metadata.extra["batched_rows"] == 2, "expected the first chunk to batch 2 rows together"
    print("PASS: rows 1+2 batched into one chunk, row 3 split off once the cap was hit")

    print("\n--- same rows, default cap (1500 chars) ---")
    chunks_default = _chunk_tabular(csv_docs)
    print(f"-> {len(chunks_default)} chunk(s) (all 3 rows should fit in one batch)")
    assert len(chunks_default) == 1
    print("PASS: default cap keeps small tables in a single chunk")


def test_full_pipeline():
    print("\n=== chunk_documents() end-to-end (needs OPENAI_API_KEY) ===")
    if not OPENAI_API_KEY:
        print("SKIPPED: no OPENAI_API_KEY in .env yet. Copy .env.example to .env and add your key to run this.")
        return

    all_docs = []
    for f in ["onboarding.pdf", "security_policy.docx", "handbook.html", "policy.csv"]:
        all_docs.extend(load_document(SAMPLE_DIR / f))

    chunks = chunk_documents(all_docs)
    print(f"{len(all_docs)} loaded Documents -> {len(chunks)} chunks")
    for c in chunks:
        print(f"  [{c.chunk_index}] doc_type={c.metadata.doc_type:5} section={c.metadata.section!r:30} "
              f"len={len(c.content)}")


if __name__ == "__main__":
    test_semantic_split()
    test_chunk_tabular()
    test_full_pipeline()
