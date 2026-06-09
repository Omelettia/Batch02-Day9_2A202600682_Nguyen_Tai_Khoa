"""Minimal stdio MCP-style server exposing PageIndex search as a tool."""

from __future__ import annotations

import json
import sys
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task8_pageindex_vectorless import pageindex_search


def handle_request(payload: dict) -> dict:
    tool = payload.get("tool")
    arguments = payload.get("arguments", {}) or {}
    if tool != "pageindex_search":
        return {"error": f"Unknown MCP tool: {tool}"}
    query = str(arguments.get("query", ""))
    top_k = int(arguments.get("top_k", 5))
    return {"results": pageindex_search(query, top_k=top_k)}


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        print(json.dumps(handle_request(payload), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

