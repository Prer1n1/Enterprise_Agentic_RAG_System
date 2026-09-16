"""Hybrid retriever — fuses dense (Chroma) and sparse (BM25) rankings
using Reciprocal Rank Fusion (RRF), not raw score averaging.

Why RRF, not averaging: cosine similarity (dense, roughly 0-1) and BM25
(sparse, unbounded and corpus-dependent) live on completely different
scales — averaging them directly means comparing apples to oranges
unless both are carefully normalized first, and that normalization step
is itself a common source of bugs (what if one retriever returns zero
results, or all-identical scores?). RRF sidesteps the problem entirely:
it only looks at RANK POSITION within each ranked list, never the raw
score, so scale mismatches simply can't happen.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from retrieval.bm25_retriever import BM25Retriever
from storage.chunk_store import ChunkStore
from storage.vector_store import similarity_search

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al.)


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict
    score: float


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """rankings: one ranked list of chunk_ids per retriever.
    score(chunk) = sum over retrievers of 1 / (k + rank_in_that_retriever).
    A chunk ranked highly by BOTH retrievers scores higher than one only
    one retriever liked — that's the "hybrid" benefit, made concrete."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, vector_store, chunk_store: ChunkStore):
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.bm25 = BM25Retriever(chunk_store)
        self._rows_by_id = {r["chunk_id"]: r for r in self.bm25.rows}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        category: str = None,
    ) -> List[RetrievedChunk]:
        """category=None searches the whole corpus; otherwise restricts
        both retrievers to one knowledge source — this is what lets the
        agent query a specific source instead of always searching
        everything."""
        dense_filter = {"category": category} if category else None
        dense_docs = similarity_search(self.vector_store, query, k=fetch_k, filter=dense_filter)
        dense_ranking = [d.metadata["chunk_id"] for d in dense_docs if d.metadata.get("chunk_id")]

        sparse_results = self.bm25.search(query, k=fetch_k, category=category)
        sparse_ranking = [chunk_id for chunk_id, _ in sparse_results]

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]

        results = []
        for chunk_id in top_ids:
            row = self._rows_by_id.get(chunk_id)
            if row is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=row["content"],
                    metadata={
                        key: row[key]
                        for key in ("source", "doc_type", "section", "page_number", "category")
                    },
                    score=fused[chunk_id],
                )
            )
        return results
