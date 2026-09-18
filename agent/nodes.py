"""LangGraph nodes: plan (route to categories) -> retrieve (fanned out in
parallel, one branch per category) -> synthesize (merge + grounded answer
with citations).
"""

from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langgraph.types import Send

from agent.planner import plan_categories
from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from retry_utils import retry_openai_call

_SYNTHESIS_MODEL = "gpt-4o-mini"

_SYNTHESIS_PROMPT = """Answer the user's question using ONLY the numbered context below.
Cite sources inline like [1], [2] matching the numbered context.
If the context doesn't contain the answer, say so directly — never invent facts.

Context:
{context}

Question: {query}"""


def plan_node(state: AgentState) -> dict:
    return {"categories": plan_categories(state["query"])}


def route_to_sources(state: AgentState) -> List[Send]:
    """Conditional edge: fans out to one retrieve_node execution PER
    category, IN PARALLEL. This is the literal implementation of
    "querying multiple knowledge sources before synthesizing" — not one
    filtered search, but genuinely separate retrieval calls per source
    that LangGraph runs concurrently and merges back together."""
    return [
        Send("retrieve_node", {"query": state["query"], "category": category})
        for category in state["categories"]
    ]


def make_retrieve_node(retriever: HybridRetriever):
    """Returns a retrieve_node bound to a specific HybridRetriever
    instance — a closure instead of a module-level global, so the graph
    can be built against different retrievers (e.g. a test double)."""

    def retrieve_node(state: dict) -> dict:
        chunks = retriever.retrieve(state["query"], top_k=5, category=state["category"])
        return {"retrieved_chunks": chunks}

    return retrieve_node


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        location = c.metadata.get("section") or f"page {c.metadata.get('page_number')}"
        lines.append(f"[{i}] (source: {c.metadata['source']}, {location})\n{c.content}")
    return "\n\n".join(lines)


@retry_openai_call
def _invoke_synthesis(llm, prompt: str):
    """Unlike plan_categories/classify_category, this has no fallback to
    degrade to — a failed synthesis call IS the failed request. Retrying a
    transient blip here is what stands between a brief network hiccup and
    a real user-facing 500. See retry_utils.py."""
    return llm.invoke(prompt)


def synthesize_node(state: AgentState) -> dict:
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {
            "answer": "I couldn't find anything in the knowledge base relevant to that question.",
            "citations": [],
        }

    # The same chunk can come back from more than one category branch
    # (e.g. it scored well in both HR and General) — dedupe before synthesis.
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique_chunks.append(c)

    context = _format_context(unique_chunks)
    llm = ChatOpenAI(model=_SYNTHESIS_MODEL, temperature=0)
    response = _invoke_synthesis(llm, _SYNTHESIS_PROMPT.format(context=context, query=state["query"]))

    citations = [
        {
            "source": c.metadata["source"],
            "section": c.metadata.get("section"),
            "category": c.metadata.get("category"),
        }
        for c in unique_chunks
    ]
    return {"answer": response.content, "citations": citations}
