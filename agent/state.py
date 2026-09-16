"""Shared state that flows through the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from retrieval.hybrid_retriever import RetrievedChunk


class AgentState(TypedDict):
    query: str
    categories: List[str]  # decided by the planner node
    # Annotated with operator.add: when multiple retrieve_node branches run
    # in PARALLEL (one per category), LangGraph needs to know to CONCATENATE
    # their results into this list, not have the last branch silently
    # overwrite the others.
    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add]
    answer: str
    citations: List[dict]
