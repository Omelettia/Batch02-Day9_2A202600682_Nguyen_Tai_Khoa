"""Readable trace logging for the Day08 supervisor workflow."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator
from uuid import uuid4
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(value: object, limit: int = 220) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class TraceEvent:
    trace_id: str
    step: int
    agent: str
    action: str
    started_at: str
    ended_at: str
    duration_ms: float
    input_summary: str
    output_summary: str
    status: str = "ok"


class TraceLogger:
    """Collects trace events and writes JSONL for later explanation/demo."""

    def __init__(self, trace_id: str | None = None, write_file: bool = True) -> None:
        self.trace_id = trace_id or str(uuid4())
        self.write_file = write_file
        self.events: list[TraceEvent] = []
        self._step = 0
        self.path = LOG_DIR / f"supervisor_trace_{self.trace_id}.jsonl"
        if write_file:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @contextmanager
    def span(self, agent: str, action: str, input_data: object = "") -> Iterator[dict[str, object]]:
        started_at = _now_iso()
        started = perf_counter()
        state: dict[str, object] = {"output": "", "status": "ok"}
        try:
            yield state
        except Exception as exc:
            state["output"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "error"
            raise
        finally:
            self._step += 1
            event = TraceEvent(
                trace_id=self.trace_id,
                step=self._step,
                agent=agent,
                action=action,
                started_at=started_at,
                ended_at=_now_iso(),
                duration_ms=round((perf_counter() - started) * 1000, 2),
                input_summary=summarize(input_data),
                output_summary=summarize(state.get("output", "")),
                status=str(state.get("status", "ok")),
            )
            self.events.append(event)
            line = json.dumps(asdict(event), ensure_ascii=False)
            print(f"[{event.step:02d}] {event.agent}.{event.action} - {event.status} ({event.duration_ms} ms)")
            if self.write_file:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def to_list(self) -> list[dict]:
        return [asdict(event) for event in self.events]
