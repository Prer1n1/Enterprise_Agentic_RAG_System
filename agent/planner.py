"""Planner (router) — the actual "agentic" decision in this project: which
knowledge source(s) does this query need? Uses an LLM with structured
output rather than hard-coded rules, because query intent is often
ambiguous in ways keyword matching misses ("what happens if I violate
this" could be HR, Security, or Legal depending on context — a keyword
list can't weigh that, an LLM can).
"""

from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ingestion.metadata_extractor import CATEGORIES

_PLANNER_MODEL = "gpt-4o-mini"


class RoutingDecision(BaseModel):
    categories: List[str] = Field(
        description=(
            f"Which of these knowledge source categories are relevant to the "
            f"query: {CATEGORIES}. Pick only what's actually needed — one "
            f"category for a focused question, more for a question that spans "
            f"domains. Never invent a category that isn't in the list."
        )
    )


def plan_categories(query: str) -> List[str]:
    """Returns the categories to query. Falls back to querying ALL known
    categories if the LLM call fails or returns something unusable — a
    routing mistake should degrade to 'search broadly', never to 'search
    nothing'."""
    try:
        llm = ChatOpenAI(model=_PLANNER_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(RoutingDecision)
        decision = structured_llm.invoke(
            f"User query: {query}\n\n"
            f"Which knowledge source categories should be searched to answer this?"
        )
        valid = [c for c in decision.categories if c in CATEGORIES]
        return valid or list(CATEGORIES)
    except Exception:
        return list(CATEGORIES)
