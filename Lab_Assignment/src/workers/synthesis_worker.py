"""Synthesis worker: creates cited answers from already-retrieved chunks."""

from __future__ import annotations

try:
    from ..task10_generation import (
        _build_prompt,
        _extractive_answer,
        _generate_with_gemini,
        _generate_with_openai,
        format_context,
        reorder_for_llm,
    )
    from ..trace_log import TraceLogger
except ImportError:
    from task10_generation import (
        _build_prompt,
        _extractive_answer,
        _generate_with_gemini,
        _generate_with_openai,
        format_context,
        reorder_for_llm,
    )
    from trace_log import TraceLogger


def synthesize_answer(
    query: str,
    chunks: list[dict],
    trace: TraceLogger,
    use_llm: bool = True,
    conversation_context: str | None = None,
) -> dict:
    with trace.span("synthesis_worker", "generate_with_citations", {"query": query, "chunks": len(chunks)}) as span:
        reordered = reorder_for_llm(chunks)
        prompt = _build_prompt(query, format_context(reordered), conversation_context)
        answer = None
        if use_llm:
            answer = _generate_with_gemini(prompt) or _generate_with_openai(prompt)
        if not answer:
            answer = _extractive_answer(query, reordered)
        span["output"] = f"answer_chars={len(answer)}, sources={len(chunks)}"
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        }
