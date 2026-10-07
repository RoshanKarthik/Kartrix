"""Local run traces (roadmap 2.4) and optional LangSmith.

Every run (a chat turn, a ``/plan`` run, a headless run) is recorded as one JSON-lines file in
``tracing.dir`` (default ``.kartrix/traces/``, which the agent itself can't read): the run's events —
agent steps, every model call (model, tokens, latency, finish reason), every tool call (target,
outcome, duration), approvals, files changed, the answer and the usage. ``kartrix trace`` lists the
recent runs and renders one as a timeline with per-agent and per-model totals and the slowest steps —
enough to see where a run spent its time and tokens without any service.

LangSmith (LangChain's tracing service) is used **only** when ``LANGSMITH_API_KEY`` is set and
``tracing.langsmith`` isn't ``off``: then LangChain's own tracer sends every model/tool span there
(project ``tracing.langsmith_project``). It sends prompts — i.e. code — to a hosted service, so it is
opt-in by having the key; the local traces never leave the machine.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from kartrix.config import settings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_SKIP = frozenset({"assistant_delta"})  # streamed pieces: the full answer is recorded once


def traces_dir() -> Path:
    return Path(settings.tracing.dir)


def enable_langsmith() -> bool:
    """Turn on LangChain's LangSmith tracer when a key is configured (and tracing isn't switched off)."""
    if settings.tracing.langsmith == "off" or not os.environ.get("LANGSMITH_API_KEY", "").strip():
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.tracing.langsmith_project)
    try:  # langsmith caches environment lookups: a value read before this point would win
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()  # type: ignore[attr-defined]
        ls_utils.get_tracer_project.cache_clear()
    except (ImportError, AttributeError):
        pass
    return True


class TraceRecorder:
    """Event subscriber: one JSON-lines file per run id (events outside a run are ignored)."""

    def __init__(self, folder: Path | None = None, keep: int | None = None) -> None:
        self.folder = folder or traces_dir()
        self.keep = settings.tracing.keep if keep is None else keep
        self._files: dict[str, IO[str]] = {}
        self._lock = threading.Lock()

    def __call__(self, event: Any) -> None:
        run_id = getattr(event, "run_id", None)
        if run_id is None or event.type in _SKIP:
            return
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self._lock:
            stream = self._files.get(run_id)
            if stream is None:
                stream = self._open(run_id)
                if stream is None:
                    return
            stream.write(line + "\n")
            stream.flush()
            if event.type == "run_finished":
                stream.close()
                del self._files[run_id]

    def _open(self, run_id: str) -> IO[str] | None:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            stream = (self.folder / f"{datetime.now():%Y%m%d-%H%M%S}-{run_id}.jsonl").open("w", encoding="utf-8")
        except OSError as e:  # a trace must never break a run
            logger.warning("Could not write the run trace", extra={"error": str(e)})
            return None
        self._files[run_id] = stream
        self._prune()
        return stream

    def _prune(self) -> None:
        files = sorted(self.folder.glob("*.jsonl"))
        for old in files[: max(0, len(files) - self.keep)]:
            try:
                old.unlink()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            for stream in self._files.values():
                stream.close()
            self._files.clear()


# ── reading traces ────────────────────────────────────────────────────


def list_traces(folder: Path | None = None) -> list[Path]:
    folder = folder or traces_dir()
    return sorted(folder.glob("*.jsonl"), reverse=True) if folder.is_dir() else []


def load(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def find(ref: str, folder: Path | None = None) -> Path | None:
    """A trace by run id (prefix), file name, or ``last``."""
    traces = list_traces(folder)
    if ref == "last":
        return traces[0] if traces else None
    return next((p for p in traces if ref in p.stem.split("-", 2)[-1] or p.name.startswith(ref)), None)


@dataclass
class _Span:
    agent: str
    start: float
    end: float | None = None
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tools: list[str] = field(default_factory=list)


@dataclass
class Summary:
    run_id: str
    kind: str
    input: str
    status: str
    seconds: float
    usage: dict[str, Any]
    spans: list[_Span]
    models: dict[str, dict[str, float]]
    slowest_tools: list[tuple[float, str, str]]
    slowest_models: list[tuple[float, str, str]]
    approvals: int
    files: list[str]
    answer: str


def summarize(events: list[dict[str, Any]]) -> Summary:
    started = next((e for e in events if e["type"] == "run_started"), {})
    finished = next((e for e in events if e["type"] == "run_finished"), {})
    t0 = started.get("ts", events[0]["ts"] if events else 0.0)
    spans: list[_Span] = []
    open_span: _Span | None = None
    models: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "ms": 0.0})
    tool_starts: dict[str, dict[str, Any]] = {}
    slow_tools: list[tuple[float, str, str]] = []
    slow_models: list[tuple[float, str, str]] = []
    files: list[str] = []
    for e in events:
        kind = e["type"]
        if kind == "agent_step" and e.get("status") == "started":
            if open_span is None or open_span.agent != e["agent"] or open_span.end is not None:
                open_span = _Span(e["agent"], e["ts"] - t0)
                spans.append(open_span)
        elif kind == "agent_step" and e.get("status") == "finished" and open_span is not None:
            open_span.end = e["ts"] - t0
        elif kind == "model_call":
            m = models[e["model"]]
            m["calls"] += 1
            m["in"] += e.get("input_tokens", 0)
            m["out"] += e.get("output_tokens", 0)
            m["ms"] += e.get("duration_ms", 0.0)
            agent = open_span.agent if open_span is not None else "agent"
            slow_models.append((e.get("duration_ms", 0.0), e["model"], agent))
            if open_span is not None:
                open_span.model_calls += 1
                open_span.tokens_in += e.get("input_tokens", 0)
                open_span.tokens_out += e.get("output_tokens", 0)
        elif kind == "tool_call_started":
            tool_starts[e.get("call_id") or ""] = e
            if open_span is not None:
                open_span.tools.append(e["tool"])
        elif kind == "tool_call_finished":
            begin = tool_starts.get(e.get("call_id") or "", {})
            label = f"{e['tool']} {(begin.get('target') or '')[:60]}".strip()
            slow_tools.append((e.get("duration_ms", 0.0), label, e.get("outcome", "")))
        elif kind == "files_changed":
            files = list(e.get("files", []))
    answer = next((e.get("text", "") for e in reversed(events) if e["type"] == "assistant_message"), "")
    end = finished.get("ts", events[-1]["ts"] if events else t0)
    for span in spans:
        if span.end is None:
            span.end = end - t0
    return Summary(
        run_id=started.get("run_id") or finished.get("run_id") or "?",
        kind=started.get("kind", "?"),
        input=started.get("input", ""),
        status=finished.get("status", "running or interrupted"),
        seconds=round(end - t0, 2),
        usage=finished.get("usage") or {},
        spans=spans,
        models=dict(models),
        slowest_tools=sorted(slow_tools, reverse=True)[:5],
        slowest_models=sorted(slow_models, reverse=True)[:3],
        approvals=sum(1 for e in events if e["type"] == "approval_requested"),
        files=files,
        answer=answer,
    )


def _tok(n: float) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def render(summary: Summary) -> str:
    u = summary.usage
    head = (
        f"run {summary.run_id} · {summary.kind} · {summary.status} in {summary.seconds:.1f} s · "
        f"{_tok(u.get('input_tokens', 0))} in / {_tok(u.get('output_tokens', 0))} out tokens · "
        f"{u.get('model_calls', 0)} model calls · {u.get('tool_calls', 0)} tool calls"
    )
    if u.get("cost_known"):
        head += f" · ${u.get('cost_usd', 0):.4f}"
    lines = [head, f"  request: {summary.input[:160]}", ""]
    lines.append("  timeline")
    for s in summary.spans:
        dur = (s.end or s.start) - s.start
        tools = _tool_counts(s.tools)
        lines.append(
            f"  {s.start:7.1f}s  {s.agent:<10} {dur:6.1f}s  {s.model_calls:>2} model call{'' if s.model_calls == 1 else 's'}  "
            f"{_tok(s.tokens_in + s.tokens_out):>6} tok" + (f"  · {tools}" if tools else "")
        )
    if summary.models:
        lines += ["", "  models"]
        for name, m in summary.models.items():
            avg = m["ms"] / m["calls"] / 1000 if m["calls"] else 0
            lines.append(
                f"    {name:<44} {int(m['calls']):>3} call{'' if m['calls'] == 1 else 's'}  {_tok(m['in'])} in / {_tok(m['out'])} out  avg {avg:.1f} s"
            )
    if summary.slowest_models:
        lines += ["", "  slowest model calls"]
        lines += [f"    {ms / 1000:6.1f} s  {name} ({agent})" for ms, name, agent in summary.slowest_models]
    if summary.slowest_tools:
        lines += ["", "  slowest tool calls"]
        lines += [f"    {ms / 1000:6.1f} s  {label}  [{outcome}]" for ms, label, outcome in summary.slowest_tools]
    if summary.approvals:
        lines.append(f"\n  approvals asked: {summary.approvals}")
    if summary.files:
        lines.append(f"  files changed: {', '.join(summary.files[:10])}")
    return "\n".join(lines)


def _tool_counts(tools: list[str]) -> str:
    counts: dict[str, int] = {}
    for t in tools:
        counts[t] = counts.get(t, 0) + 1
    return ", ".join(f"{t}×{n}" if n > 1 else t for t, n in counts.items())


def render_list(paths: list[Path], limit: int = 15) -> str:
    rows = []
    for path in paths[:limit]:
        s = summarize(load(path))
        when = datetime.strptime(path.stem[:15], "%Y%m%d-%H%M%S") if path.stem[:15].replace("-", "").isdigit() else None
        rows.append(f"{when:%m-%d %H:%M} " if when else "            ")
        u = s.usage
        rows[-1] += (
            f"{s.run_id:<13} {s.kind:<5} {s.status:<10} {s.seconds:7.1f} s  "
            f"{_tok(u.get('input_tokens', 0) + u.get('output_tokens', 0)):>6} tok  {s.input[:60]}"
        )
    return "\n".join(rows) if rows else "No traces yet — they are written for every run."


def stats(paths: list[Path]) -> str:
    """Usage and cost over many runs: outcomes, time, tokens and cost — overall and per model."""
    runs = [summarize(load(p)) for p in paths]
    if not runs:
        return "No traces yet — they are written for every run."
    by_status: dict[str, int] = defaultdict(int)
    models: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "ms": 0.0})
    tokens = cost = seconds = 0.0
    cost_known = True
    for r in runs:
        by_status[r.status] += 1
        seconds += r.seconds
        tokens += r.usage.get("input_tokens", 0) + r.usage.get("output_tokens", 0)
        cost += r.usage.get("cost_usd", 0.0)
        cost_known = cost_known and bool(r.usage.get("cost_known", False))
        for name, m in r.models.items():
            for key in ("calls", "in", "out", "ms"):
                models[name][key] += m[key]
    n = len(runs)
    lines = [
        f"{n} runs · " + " · ".join(f"{k} {v}" for k, v in sorted(by_status.items(), key=lambda kv: -kv[1])),
        f"time: {seconds:.0f} s total, {seconds / n:.1f} s per run · tokens: {_tok(tokens)} total, "
        f"{_tok(tokens / n)} per run · cost: "
        + (f"${cost:.4f}" if cost_known else f"≥ ${cost:.4f} (some models unpriced)"),
        "",
        "  per model",
    ]
    for name, m in sorted(models.items(), key=lambda kv: -kv[1]["calls"]):
        avg = m["ms"] / m["calls"] / 1000 if m["calls"] else 0
        lines.append(
            f"    {name:<44} {int(m['calls']):>5} calls  {_tok(m['in'])} in / {_tok(m['out'])} out  avg {avg:.1f} s"
        )
    return "\n".join(lines)


def cli(ref: str | None, as_json: bool = False, show_stats: bool = False) -> int:
    """``kartrix trace [run]``: list recent runs, or show one; ``--stats``: usage and cost over all runs."""
    if show_stats:
        print(stats(list_traces()))
        return 0
    if ref is None:
        print(render_list(list_traces()))
        return 0
    path = find(ref)
    if path is None:
        print(f"No trace matches {ref!r} (see `kartrix trace`).")
        return 1
    if as_json:
        print(path.read_text(encoding="utf-8"), end="")
        return 0
    print(render(summarize(load(path))))
    return 0


__all__ = ["TraceRecorder", "cli", "enable_langsmith", "find", "list_traces", "load", "render", "summarize"]
