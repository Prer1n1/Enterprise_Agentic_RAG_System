"""Shared state that flows through the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from retrieval.hybrid_retriever import RetrievedChunk


class AgentState(TypedDict):
    query: str
    # Access control (api/app.py): which categories the CALLER's API key is
    # allowed to query, or None for no restriction (the admin key, and the
    # CLI which never sets this key at all — main.py's graph.invoke() calls
    # omit it, and .get() on a plain dict returns None for a missing key,
    # so CLI usage stays fully unrestricted with zero changes needed there).
    allowed_categories: Optional[List[str]]
    categories: List[str]  # decided by the planner node, then access-filtered
    # True if plan_node had to narrow the router's categories because of
    # allowed_categories — lets synthesize_node tell "access-restricted"
    # apart from "genuinely nothing relevant in the corpus."
    access_restricted: bool
    # Annotated with operator.add: when multiple retrieve_node branches run
    # in PARALLEL (one per category), LangGraph needs to know to CONCATENATE
    # their results into this list, not have the last branch silently
    # overwrite the others.
    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add]
    answer: str
    citations: List[dict]
