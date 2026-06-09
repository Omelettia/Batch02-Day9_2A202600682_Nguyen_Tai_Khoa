"""Supervisor orchestration for the Day08 RAG pipeline.

This module upgrades the Day08 single pipeline into a small multi-agent flow:
supervisor -> retrieval worker -> optional MCP tool worker -> synthesis worker.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    from .trace_log import TraceLogger
    from .workers.retrieval_worker import retrieve_sources
    from .workers.synthesis_worker import synthesize_answer
    from .workers.tool_worker import pageindex_mcp_search
except ImportError:
    from trace_log import TraceLogger
    from workers.retrieval_worker import retrieve_sources
    from workers.synthesis_worker import synthesize_answer
    from workers.tool_worker import pageindex_mcp_search


def _top_score(chunks: list[dict]) -> float:
    return float(chunks[0].get("score", 0.0)) if chunks else 0.0


def run_supervised_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    use_reranking: bool = True,
    use_llm: bool = True,
    conversation_context: str | None = None,
    force_mcp: bool = False,
) -> dict:
    trace = TraceLogger()
    with trace.span("supervisor", "receive_task", {"query": query}) as span:
        span["output"] = f"trace_id={trace.trace_id}"

    chunks = retrieve_sources(query, top_k, score_threshold, use_reranking, trace)

    needs_tool = force_mcp or not chunks or _top_score(chunks) < score_threshold
    with trace.span(
        "supervisor",
        "route_worker",
        {"top_score": _top_score(chunks), "threshold": score_threshold, "force_mcp": force_mcp},
    ) as span:
        span["output"] = "tool_worker" if needs_tool else "synthesis_worker"

    if needs_tool:
        tool_chunks = pageindex_mcp_search(query, top_k, trace)
        if tool_chunks:
            chunks = tool_chunks

    result = synthesize_answer(query, chunks, trace, use_llm=use_llm, conversation_context=conversation_context)
    with trace.span("supervisor", "finalize", {"sources": len(result.get("sources", []))}) as span:
        span["output"] = f"answer_chars={len(result.get('answer', ''))}"

    return {**result, "trace_id": trace.trace_id, "trace": trace.to_list(), "trace_path": str(trace.path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day08 supervised RAG workflow.")
    parser.add_argument("query", nargs="?", default="Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện nào?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--force-mcp", action="store_true")
    args = parser.parse_args()

    result = run_supervised_rag(
        args.query,
        top_k=args.top_k,
        score_threshold=args.threshold,
        use_reranking=not args.no_rerank,
        use_llm=not args.no_llm,
        force_mcp=args.force_mcp,
    )
    print("\nANSWER\n" + "=" * 60)
    print(result["answer"])
    print("\nTRACE")
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    print(f"\nTrace file: {result['trace_path']}")


if __name__ == "__main__":
    main()


