"""Retrieval worker: wraps the existing Task 9 retrieval pipeline."""

from __future__ import annotations

try:
    from ..task9_retrieval_pipeline import retrieve
    from ..trace_log import TraceLogger
except ImportError:
    from task9_retrieval_pipeline import retrieve
    from trace_log import TraceLogger


def retrieve_sources(
    query: str,
    top_k: int,
    score_threshold: float,
    use_reranking: bool,
    trace: TraceLogger,
) -> list[dict]:
    with trace.span(
        "retrieval_worker",
        "hybrid_retrieve",
        {"query": query, "top_k": top_k, "threshold": score_threshold, "rerank": use_reranking},
    ) as span:
        results = retrieve(
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranking=use_reranking,
        )
        top_score = float(results[0].get("score", 0.0)) if results else 0.0
        span["output"] = f"{len(results)} chunks, top_score={top_score:.3f}, source={results[0].get('source', 'none') if results else 'none'}"
        return results
