"""Runs the full agent pipeline over the curated eval set and scores it
per pipeline stage — router, retrieval, reranking, generation — not just
the final answer. Real, library-provided metrics wherever one exists
(RAGAS covers most of this pipeline directly); only the two checks with
no RAGAS equivalent (Routing Accuracy, Citation Correctness) are
hand-written, and both are small, mechanical, deterministic checks against
this project's own domain concepts (its category taxonomy, its citation
numbering) — not a reimplementation of anything RAGAS already does.

Stage -> metric -> RAGAS class (or "custom" with the reason):
  Router      -> Routing Accuracy         -> custom (domain-specific: this
                                              project's own category
                                              taxonomy, no general metric
                                              fits)
  Retrieval   -> Context Recall           -> LLMContextRecall
              -> Context Precision        -> LLMContextPrecisionWithReference
  Reranking   -> Ranking quality          -> custom comparison: Context
                                              Precision WITH vs WITHOUT
                                              Cohere reranking, same
                                              question/categories, isolating
                                              reranking's own effect. A
                                              THIRD column adds Laya's own
                                              relevance-scoring reranker
                                              (laya_classifier.laya_rerank)
                                              as a comparison point ONLY —
                                              evaluation-only, not wired
                                              into the live retrieval path.
                                              See docs/design-decisions.md
                                              ("Can Laya replace Cohere?").
  Generation  -> Answer correctness       -> AnswerCorrectness
              -> Groundedness             -> Faithfulness (this doubles as
                                              hallucination detection — see
                                              hallucination_guardrail.py)
              -> Answer relevancy         -> ResponseRelevancy
              -> Citation correctness     -> custom (mechanical: does every
                                              [n] marker in the answer text
                                              reference a real citation?)

Deliberately NOT implemented as eval-set metrics, with reasoning (see
docs/design-decisions.md "Guardrails — completing the set" for the fuller
version of this same reasoning):
  Completeness      -> this eval set's questions are single-fact lookups;
                        "completeness" isn't meaningfully distinguishable
                        from correctness until the eval set has genuinely
                        multi-part questions. Adding a hollow metric that
                        can't discriminate would be worse than not having
                        one.
  Tone              -> low-value for an internal tool answering terse,
                        factual policy questions from a constrained,
                        grounded prompt — there's little tone variance to
                        measure.
  Safety            -> covered by the guardrails already built (prompt
                        injection defense, PII redaction, access control)
                        rather than a separate eval-set metric; a content-
                        moderation classifier was assessed and deliberately
                        not built (see the Guardrails design-log section).
  Policy compliance -> this IS groundedness/Faithfulness under a different
                        name for a platform whose only source of truth is
                        the ingested policy documents — a separate metric
                        would just be re-deriving the same signal.

API note: ragas's `llm_factory`/`embedding_factory` (the path its own
deprecation warnings recommend) throws `AttributeError` with these
specific metric classes in this installed version (0.3.9) — verified by
testing both paths directly. LangchainLLMWrapper/LangchainEmbeddingsWrapper
are used here because they're what actually works, despite being flagged
deprecated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agent.graph import build_agent_graph
from config import EMBEDDING_MODEL
from evaluation.eval_dataset import EVAL_CASES, EvalCase
from hallucination_guardrail import HALLUCINATION_THRESHOLD
from laya_classifier import laya_rerank
from retrieval.hybrid_retriever import HybridRetriever

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class EvalRunResult:
    question: str
    answer: str
    reference_answer: str
    routing_accuracy: bool
    faithfulness: float
    answer_relevancy: float
    answer_correctness: float
    context_precision: float
    context_recall: float
    citation_correctness: bool
    likely_hallucination: bool
    # None if reranking wasn't available for this run (no COHERE_API_KEY) —
    # see _reranking_impact()'s docstring.
    context_precision_with_rerank: float
    context_precision_without_rerank: float
    # Laya's OWN reranking signal (laya_classifier.laya_rerank), applied to
    # the same RRF-only candidate pool as context_precision_without_rerank —
    # isolates Laya's reranking quality from Cohere's, both measured
    # against the identical un-reranked baseline. See _laya_reranking_impact().
    context_precision_with_laya_rerank: float


def _routing_accuracy(actual_categories: List[str], expected_categories: List[str]) -> bool:
    """Intersection-based, not exact-match: the router is deliberately
    allowed to be broader than strictly necessary (see agent/planner.py's
    own "more for a question that spans domains" guidance) — a routing
    decision is only WRONG if it misses every expected category, not if
    it includes extras. See EvalCase.expected_categories."""
    return bool(set(actual_categories) & set(expected_categories))


def _citation_correctness(answer: str, citation_count: int) -> bool:
    """Structural, not semantic: does every [n] marker in the answer text
    reference a real citation index? A mechanical LLM error mode
    (citing [4] when only 3 sources were actually retrieved) that
    Faithfulness/Relevancy don't check, since they judge the CONTENT of
    the answer, not whether its own citation numbering is internally
    consistent."""
    markers = {int(m) for m in _CITATION_MARKER_RE.findall(answer)}
    if not markers:
        return citation_count == 0
    return all(1 <= m <= citation_count for m in markers)


def _dedupe_contexts(chunks) -> List[str]:
    """Same de-dupe synthesize_node applies, so context fed to RAGAS
    matches what the LLM actually saw when writing the answer."""
    seen = set()
    contexts = []
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            contexts.append(chunk.content)
    return contexts


def _reranking_impact(
    retriever: HybridRetriever,
    case: EvalCase,
    categories: List[str],
    precision_metric: LLMContextPrecisionWithReference,
) -> Tuple[float, float]:
    """Context Precision WITH vs WITHOUT Cohere reranking, for the SAME
    question and the SAME routed categories — isolating reranking's own
    effect from routing's effect (a different set of categories would
    confound the comparison). Reuses HybridRetriever.retrieve()'s own
    use_reranker toggle (retrieval/hybrid_retriever.py) rather than
    reimplementing retrieval here."""

    def _contexts_for(use_reranker: bool) -> List[str]:
        all_chunks = []
        for category in categories:
            all_chunks.extend(
                retriever.retrieve(case.question, top_k=5, category=category, use_reranker=use_reranker)
            )
        return _dedupe_contexts(all_chunks)

    with_rerank = SingleTurnSample(
        user_input=case.question, retrieved_contexts=_contexts_for(True), reference=case.reference_answer
    )
    without_rerank = SingleTurnSample(
        user_input=case.question, retrieved_contexts=_contexts_for(False), reference=case.reference_answer
    )
    return (
        precision_metric.single_turn_score(with_rerank),
        precision_metric.single_turn_score(without_rerank),
    )


def _laya_reranking_impact(
    retriever: HybridRetriever,
    case: EvalCase,
    categories: List[str],
    precision_metric: LLMContextPrecisionWithReference,
) -> float:
    """Context Precision with Laya's OWN relevance-scoring reranker
    (laya_classifier.laya_rerank) applied to the same RRF-only candidate
    pool _reranking_impact() uses for its "without rerank" baseline —
    same isolation principle: identical question/categories, only the
    reranking step itself differs, so this and
    context_precision_without_rerank are directly comparable, and both
    are directly comparable to context_precision_with_rerank (Cohere).

    A wider top_k=10 pool (vs. the final top 5) gives Laya's reranker
    actual candidates to discriminate between — reranking a pool that's
    already been trimmed to the final size would just reorder the same 5
    items Cohere/RRF already settled on, not test reranking quality.
    Falls back to plain RRF order if Laya is unavailable or fails, same
    graceful-degradation contract as the rest of this pilot."""
    all_chunks = []
    for category in categories:
        all_chunks.extend(
            retriever.retrieve(case.question, top_k=10, category=category, use_reranker=False)
        )
    candidates = _dedupe_contexts(all_chunks)
    if not candidates:
        contexts = []
    else:
        reranked = laya_rerank(case.question, candidates, top_n=5)
        contexts = [candidates[i] for i, _ in reranked] if reranked is not None else candidates[:5]
    sample = SingleTurnSample(user_input=case.question, retrieved_contexts=contexts, reference=case.reference_answer)
    return precision_metric.single_turn_score(sample)


def run_evaluation(retriever: HybridRetriever) -> List[EvalRunResult]:
    graph = build_agent_graph(retriever)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    precision_metric = LLMContextPrecisionWithReference(llm=llm)

    agent_outputs = []
    samples = []
    rerank_scores = []
    laya_rerank_scores = []
    for case in EVAL_CASES:
        result = graph.invoke({"query": case.question})
        contexts = _dedupe_contexts(result["retrieved_chunks"])

        agent_outputs.append((case, result))
        samples.append(
            SingleTurnSample(
                user_input=case.question,
                retrieved_contexts=contexts,
                response=result["answer"],
                reference=case.reference_answer,
            )
        )
        rerank_scores.append(_reranking_impact(retriever, case, result["categories"], precision_metric))
        laya_rerank_scores.append(
            _laya_reranking_impact(retriever, case, result["categories"], precision_metric)
        )

    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            AnswerCorrectness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
    )
    scores = ragas_result.to_pandas()

    results = []
    for i, (case, result) in enumerate(agent_outputs):
        row = scores.iloc[i]
        faithfulness_score = float(row["faithfulness"])
        with_rerank_score, without_rerank_score = rerank_scores[i]
        results.append(
            EvalRunResult(
                question=case.question,
                answer=result["answer"],
                reference_answer=case.reference_answer,
                routing_accuracy=_routing_accuracy(result["categories"], case.expected_categories),
                faithfulness=faithfulness_score,
                answer_relevancy=float(row["answer_relevancy"]),
                answer_correctness=float(row["answer_correctness"]),
                context_precision=float(row["llm_context_precision_with_reference"]),
                context_recall=float(row["context_recall"]),
                citation_correctness=_citation_correctness(result["answer"], len(result["citations"])),
                likely_hallucination=faithfulness_score < HALLUCINATION_THRESHOLD,
                context_precision_with_rerank=with_rerank_score,
                context_precision_without_rerank=without_rerank_score,
                context_precision_with_laya_rerank=laya_rerank_scores[i],
            )
        )
    return results


def print_report(results: List[EvalRunResult]) -> None:
    print("=== Router ===")
    print(f"{'Question':<55} Routing")
    print("-" * 70)
    for r in results:
        print(f"{r.question[:53]:<55} {'OK' if r.routing_accuracy else 'MISROUTED'}")

    print("\n=== Retrieval + Reranking ===")
    print(
        f"{'Question':<40} {'CtxPrec':>7} {'CtxRec':>6} {'Prec w/Cohere':>13} "
        f"{'Prec w/Laya':>11} {'Prec w/o rerank':>15}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.question[:38]:<40} {r.context_precision:>7.2f} {r.context_recall:>6.2f} "
            f"{r.context_precision_with_rerank:>13.2f} {r.context_precision_with_laya_rerank:>11.2f} "
            f"{r.context_precision_without_rerank:>15.2f}"
        )

    print("\n=== Generation ===")
    print(f"{'Question':<45} {'Faith':>6} {'Relev':>6} {'Correct':>8} {'Citations':>10}  Flag")
    print("-" * 95)
    for r in results:
        flag = "LIKELY HALLUCINATION" if r.likely_hallucination else ""
        citation_flag = "OK" if r.citation_correctness else "BAD REFS"
        print(
            f"{r.question[:43]:<45} {r.faithfulness:>6.2f} {r.answer_relevancy:>6.2f} "
            f"{r.answer_correctness:>8.2f} {citation_flag:>10}  {flag}"
        )

    n = len(results)
    routing_acc_pct = 100 * sum(r.routing_accuracy for r in results) / n
    citation_ok_pct = 100 * sum(r.citation_correctness for r in results) / n
    print("\n=== Summary (averages) ===")
    print(f"Routing accuracy:              {routing_acc_pct:.0f}%")
    print(f"Context precision (w/ ref):    {sum(r.context_precision for r in results) / n:.2f}")
    print(f"Context recall:                {sum(r.context_recall for r in results) / n:.2f}")
    print(f"Context precision w/ Cohere:   {sum(r.context_precision_with_rerank for r in results) / n:.2f}")
    print(f"Context precision w/ Laya:     {sum(r.context_precision_with_laya_rerank for r in results) / n:.2f}")
    print(f"Context precision w/o rerank:  {sum(r.context_precision_without_rerank for r in results) / n:.2f}")
    print(f"Faithfulness (groundedness):   {sum(r.faithfulness for r in results) / n:.2f}")
    print(f"Answer relevancy:              {sum(r.answer_relevancy for r in results) / n:.2f}")
    print(f"Answer correctness:            {sum(r.answer_correctness for r in results) / n:.2f}")
    print(f"Citation correctness:          {citation_ok_pct:.0f}%")
    print(f"Likely hallucinations:         {sum(r.likely_hallucination for r in results)}/{n}")
