"""Runs the full agent pipeline over the curated eval set and scores it
with real RAGAS metrics:

  - Faithfulness: is the answer grounded in the retrieved context? This
    doubles as hallucination detection — below HALLUCINATION_THRESHOLD,
    an answer is flagged as a likely hallucination. Not a separate
    hand-rolled mechanism: a low faithfulness score IS what a hallucination
    looks like (claims not supported by the retrieved context).
  - ResponseRelevancy: does the answer actually address the question asked?
  - LLMContextPrecisionWithoutReference: did retrieval fetch useful context?

API note: ragas's `llm_factory`/`embedding_factory` (the path its own
deprecation warnings recommend) throws `AttributeError` with these
specific metric classes in this installed version (0.3.9) — verified by
testing both paths directly. LangchainLLMWrapper/LangchainEmbeddingsWrapper
are used here because they're what actually works, despite being flagged
deprecated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

from agent.graph import build_agent_graph
from config import EMBEDDING_MODEL
from evaluation.eval_dataset import EVAL_CASES
from hallucination_guardrail import HALLUCINATION_THRESHOLD
from retrieval.hybrid_retriever import HybridRetriever


@dataclass
class EvalRunResult:
    question: str
    answer: str
    reference_answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    likely_hallucination: bool


def run_evaluation(retriever: HybridRetriever) -> List[EvalRunResult]:
    graph = build_agent_graph(retriever)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))

    agent_outputs = []
    samples = []
    for case in EVAL_CASES:
        result = graph.invoke({"query": case.question})

        # Same de-dupe synthesize_node applies, so the context RAGAS scores
        # against matches what the LLM actually saw when writing the answer.
        seen = set()
        contexts = []
        for chunk in result["retrieved_chunks"]:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                contexts.append(chunk.content)

        agent_outputs.append((case, result["answer"]))
        samples.append(
            SingleTurnSample(
                user_input=case.question,
                retrieved_contexts=contexts,
                response=result["answer"],
                reference=case.reference_answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=llm,
        embeddings=embeddings,
    )
    scores = ragas_result.to_pandas()

    results = []
    for i, (case, answer) in enumerate(agent_outputs):
        row = scores.iloc[i]
        faithfulness_score = float(row["faithfulness"])
        results.append(
            EvalRunResult(
                question=case.question,
                answer=answer,
                reference_answer=case.reference_answer,
                faithfulness=faithfulness_score,
                answer_relevancy=float(row["answer_relevancy"]),
                context_precision=float(row["llm_context_precision_without_reference"]),
                likely_hallucination=faithfulness_score < HALLUCINATION_THRESHOLD,
            )
        )
    return results


def print_report(results: List[EvalRunResult]) -> None:
    print(f"{'Question':<55} {'Faith':>6} {'Relev':>6} {'CtxPrec':>8}  Flag")
    print("-" * 90)
    for r in results:
        flag = "LIKELY HALLUCINATION" if r.likely_hallucination else ""
        print(f"{r.question[:53]:<55} {r.faithfulness:>6.2f} {r.answer_relevancy:>6.2f} {r.context_precision:>8.2f}  {flag}")

    n = len(results)
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_relev = sum(r.answer_relevancy for r in results) / n
    avg_prec = sum(r.context_precision for r in results) / n
    print("-" * 90)
    print(f"{'AVERAGE':<55} {avg_faith:>6.2f} {avg_relev:>6.2f} {avg_prec:>8.2f}")
