"""Cross-encoder reranking via Cohere's Rerank API.

Why reranking at all: Reciprocal Rank Fusion (hybrid_retriever.py) only
ever looks at each retriever's RANK POSITION, never the actual semantic
relevance of a candidate to the query — it's good at not missing a
relevant chunk, but not great at ordering the top few precisely. A
reranker scores each (query, candidate) pair directly, catching cases
where a chunk ranks decently on hybrid fusion but is a meaningfully
weaker match than something just below it.

Why Cohere specifically, not a local cross-encoder or an LLM-prompted
rerank: it's a model purpose-built for this one job (not a repurposed
embedding or chat model), needs no local GPU or model download, and is
the answer most commonly expected when "reranking" comes up in a RAG
system design discussion.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cohere

from config import COHERE_API_KEY
from retry_utils import retry_cohere_call

logger = logging.getLogger(__name__)

_RERANK_MODEL = "rerank-v3.5"


@retry_cohere_call
def _call_rerank(client: cohere.ClientV2, query: str, documents: List[str], top_n: int):
    return client.rerank(model=_RERANK_MODEL, query=query, documents=documents, top_n=top_n)


def rerank(query: str, documents: List[str], top_n: int) -> Optional[List[Tuple[int, float]]]:
    """Returns (index_into_documents, relevance_score) pairs, best-first,
    truncated to top_n. Returns None if reranking isn't available or
    fails for ANY reason (no COHERE_API_KEY configured, network error,
    exhausted retries) — the caller falls back to its pre-rerank ordering
    instead of breaking retrieval outright. Same graceful-degradation
    philosophy as the classifier/router LLM fallbacks elsewhere in this
    project: an optional quality improvement should never become a hard
    dependency for the feature it's improving."""
    if not COHERE_API_KEY or not documents:
        return None
    try:
        client = cohere.ClientV2(api_key=COHERE_API_KEY)
        response = _call_rerank(client, query, documents, top_n)
        return [(result.index, result.relevance_score) for result in response.results]
    except Exception:
        logger.exception("rerank_failed")
        return None
