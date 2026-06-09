"""MCP client for the Day08 supervisor workflow.

The client uses a small stdio JSON bridge implemented in src.mcp_server. This keeps
MCP/tool invocation as an external capability boundary while remaining runnable in
minimal local environments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def call_pageindex_search(query: str, top_k: int = 5, timeout: int = 60) -> list[dict]:
    payload = {"tool": "pageindex_search", "arguments": {"query": query, "top_k": top_k}}
    proc = subprocess.run(
        [sys.executable, "-m", "src.mcp_server"],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "MCP bridge failed")
    json_line = next(
        (line for line in reversed((proc.stdout or "").splitlines()) if line.strip().startswith("{")),
        "{}",
    )
    response = json.loads(json_line)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response.get("results", [])


