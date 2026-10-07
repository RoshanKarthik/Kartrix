"""``kartrix run --spec <file>``: one non-interactive run in the current directory (K1).

Nothing is read from the terminal: approvals come from the spec's policy, a plan is approved as
proposed (and recorded). Progress goes to stderr (``--quiet`` turns it off), every event can be
written as JSON lines (``events``), and a JSON report goes to ``report`` or stdout.

Exit codes: 0 completed · 1 failed (a task failed, or the agent raised) · 2 stopped (budget
limit or kill switch) · 3 could not start (invalid spec, configuration, services).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from kartrix.headless.spec import RunSpec, SpecError, load_spec

EXIT_CODES = {"completed": 0, "failed": 1, "error": 1, "stopped": 2, "not_started": 3}


class JsonlWriter:
    """Event subscriber writing one JSON object per line (thread-safe)."""

    def __init__(self, stream: IO[str]) -> None:
        self.stream = stream
        self._lock = threading.Lock()

    def __call__(self, event: Any) -> None:
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


def _limits(spec: RunSpec) -> Any:
    from kartrix.config import settings

    base = settings.budgets.plan if spec.mode == "plan" else settings.budgets.turn
    if spec.budget is None:
        return base
    return base.model_copy(update={k: getattr(spec.budget, k) for k in spec.budget.model_fields_set})


def build_report(spec: RunSpec, spec_path: str, outcome: Any, events: list[Any], started: float) -> dict[str, Any]:
    from kartrix.config import settings
    from kartrix.sandbox.manager import get_sandbox

    calls = [e for e in events if e.type == "tool_call_finished"]
    requested = {e.tool_call_id: e for e in events if e.type == "approval_requested"}
    approvals = [
        {**requested[e.tool_call_id].model_dump(include={"tool", "command", "category", "reason", "task_key"}),
         "decision": e.decision, "by": e.by, "message": e.message}
        for e in events if e.type == "approval_resolved" and e.tool_call_id in requested
    ]  # fmt: skip
    files = sorted({f for e in events if e.type == "files_changed" for f in e.files})
    report: dict[str, Any] = {
        "spec": spec_path,
        "task": spec.task,
        "mode": spec.mode,
        "permissions": spec.permissions,
        "status": outcome.status if outcome else "not_started",
        "exit_code": EXIT_CODES[outcome.status if outcome else "not_started"],
        "detail": outcome.detail if outcome else None,
        "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
        "duration_s": round(time.time() - started, 2),
        "workspace": str(Path.cwd()),
        "model": f"{settings.llm.provider}/{settings.llm.model}",
        "sandbox": (sb.name if (sb := get_sandbox()) else None),
        "budget": _limits(spec).model_dump(),
        "usage": outcome.usage.model_dump() if outcome else None,
        "answer": outcome.answer if outcome else None,
        "cached": outcome.cached if outcome else False,
        "plan": None,
        "tool_calls": {
            "total": len(calls),
            "by_tool": dict(Counter(e.tool for e in calls)),
            "by_outcome": dict(Counter(e.outcome for e in calls)),
        },
        "approvals": approvals,
        "files_changed": files,
        "events": len(events),
    }
    if outcome is not None and outcome.plan is not None:
        p = outcome.plan
        report["plan"] = {
            "project_id": p.project_id,
            "status": p.status,
            "resumed": p.resumed,
            "plan": p.plan,
            "tasks": [
                {
                    k: (t.get(k) or "")[:2000] if k in ("result", "error") else t.get(k)
                    for k in ("id", "title", "task_type", "status", "retry_count", "max_retries", "error", "result")
                }
                for t in p.tasks
            ],
        }
    return report


async def _run(spec: RunSpec) -> Any:
    from kartrix.core.interaction import approve_plan_as_is
    from kartrix.core.session import CoreSession, StartOptions
    from kartrix.headless.policy import PolicyApprover
    from kartrix.security.permissions import set_mode

    set_mode(spec.permissions)
    options = StartOptions(
        session_id=str(uuid.uuid4()),  # a fresh conversation; the REPL's current session isn't touched
        watch=False,
        mcp=False,
        semantic_cache=spec.semantic_cache,
        resume_pending=False,
        index="wait",  # runs are measured: search sees the whole index from the first call
    )
    core = await CoreSession.start(PolicyApprover(spec.approvals), approve_plan_as_is, options)
    try:
        for name in spec.mcp:
            assert core.mcp is not None  # noqa: S101 — created by start()
            await core.mcp.connect(name)
        if spec.mcp:
            core.rebuild_agent()
        if spec.mode == "ask":
            return await core.ask(spec.task, _limits(spec))
        return await core.plan(spec.task, _limits(spec), resume="same_goal", plan_only=spec.plan_only)
    finally:
        await core.close()


def run_headless(spec_path: str, report: str | None = None, events: str | None = None, quiet: bool = False) -> int:
    started = time.time()
    try:
        spec = load_spec(spec_path)
    except SpecError as e:
        print(f"kartrix run: {e}", file=sys.stderr)
        return EXIT_CODES["not_started"]
    report_path = report or spec.report
    events_path = events or spec.events

    from rich.console import Console

    from kartrix.core.events import bus, collecting
    from kartrix.ui.console import ConsoleRenderer

    unsubscribe = []
    events_file = open(events_path, "w", encoding="utf-8") if events_path else None
    if events_file is not None:
        unsubscribe.append(bus.subscribe(JsonlWriter(events_file)))
    if not quiet:
        unsubscribe.append(bus.subscribe(ConsoleRenderer(Console(stderr=True), show_tool_calls=True)))

    outcome = None
    error: str | None = None
    with collecting() as seen:
        try:
            outcome = asyncio.run(_run(spec))
        except KeyboardInterrupt:
            error = "interrupted"
        except Exception as e:  # configuration, missing key, database down, …
            error = f"{type(e).__name__}: {e}"
        finally:
            for fn in unsubscribe:
                fn()
            if events_file is not None:
                events_file.close()

    data = build_report(spec, spec_path, outcome, list(seen), started)
    if error:
        data["detail"] = error
        if error == "interrupted":
            data["status"], data["exit_code"] = "stopped", EXIT_CODES["stopped"]
        print(f"kartrix run: could not complete: {error}", file=sys.stderr)
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(text + os.linesep, encoding="utf-8")
    else:
        print(text)
    return int(data["exit_code"])
