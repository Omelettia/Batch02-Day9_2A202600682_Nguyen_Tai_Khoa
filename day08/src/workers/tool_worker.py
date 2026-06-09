"""Tool worker: invokes external PageIndex capability through the MCP bridge."""

from __future__ import annotations

try:
    from ..mcp_client import call_pageindex_search
    from ..trace_log import TraceLogger
except ImportError:
    from mcp_client import call_pageindex_search
    from trace_log import TraceLogger


def pageindex_mcp_search(query: str, top_k: int, trace: TraceLogger) -> list[dict]:
    with trace.span("tool_worker", "mcp_pageindex_search", {"query": query, "top_k": top_k}) as span:
        results = call_pageindex_search(query, top_k=top_k)
        span["output"] = f"{len(results)} chunks via MCP PageIndex bridge"
        return results
