"""Sparse / keyword retrieval — BM25 over the chunk store's persisted
text. This is the other half of hybrid retrieval: BM25 catches exact
term matches (policy IDs, specific names, error codes) that a semantic
embedding can blur together into "similar meaning"; dense embedding
search catches paraphrases and synonyms that pure keyword matching
misses. Neither alone is enough — that's the reason hybrid retrieval
exists at all, not just a box to check.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from storage.chunk_store import ChunkStore

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_store: ChunkStore):
        self.rows = chunk_store.get_all()
        corpus = [_tokenize(r["content"]) for r in self.rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 10, category: Optional[str] = None) -> List[Tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] ranked best first.

        category filters AFTER scoring against the full corpus, not by
        rebuilding a smaller per-category index — that keeps BM25's IDF
        statistics (how rare a term is) computed over the real corpus
        size instead of being skewed by a tiny per-category subset.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.rows, scores), key=lambda pair: pair[1], reverse=True)
        if category:
            ranked = [pair for pair in ranked if pair[0]["category"] == category]
        return [(row["chunk_id"], score) for row, score in ranked[:k]]
