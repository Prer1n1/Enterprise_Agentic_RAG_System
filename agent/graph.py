"""Builds the LangGraph workflow:

    START -> plan_node -> [retrieve_node x N, parallel] -> synthesize_node -> END

plan_node decides WHICH knowledge sources are relevant (the agentic
decision). Each selected category gets its own retrieve_node execution,
fanned out in parallel via the Send API. synthesize_node merges every
branch's results and generates one grounded, cited answer.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import make_retrieve_node, plan_node, route_to_sources, synthesize_node
from agent.state import AgentState
from retrieval.hybrid_retriever import HybridRetriever


def build_agent_graph(retriever: HybridRetriever):
    graph = StateGraph(AgentState)

    graph.add_node("plan_node", plan_node)
    graph.add_node("retrieve_node", make_retrieve_node(retriever))
    graph.add_node("synthesize_node", synthesize_node)

    graph.add_edge(START, "plan_node")
    # "synthesize_node" is a valid target here too, not just "retrieve_node"
    # — route_to_sources routes straight there when categories is empty
    # (an off-topic query, or an access-scoped query with zero category
    # overlap). Real bug this fixed: with only "retrieve_node" listed,
    # an empty Send list was a graph dead end — retrieve_node never ran,
    # so synthesize_node never ran either, and the final state was
    # missing "answer"/"citations" entirely (a live KeyError on
    # result["answer"], not a graceful "nothing found" response).
    graph.add_conditional_edges("plan_node", route_to_sources, ["retrieve_node", "synthesize_node"])
    graph.add_edge("retrieve_node", "synthesize_node")
    graph.add_edge("synthesize_node", END)

    return graph.compile()
