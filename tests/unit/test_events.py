"""Event stream (C1): bus, run tagging, JSON round trip, and the REPL renderer."""

from __future__ import annotations

import asyncio
import json
import threading

from rich.console import Console

from kartrix.core import events as ev
from kartrix.core.events import EVENT_ADAPTER, bus, collecting, emit, notice, run_scope
from kartrix.ui.console import ConsoleRenderer

ALL_EVENTS: list[ev.Event] = [
    ev.Notice(text="hi", level="warning"),
    ev.RunStarted(kind="ask", input="q", session_id="s"),
    ev.RunFinished(kind="plan", status="stopped", detail="budget", usage=ev.Usage(input_tokens=3, cost_known=False)),
    ev.AssistantMessage(text="answer", cached=True),
    ev.ToolCallStarted(call_id="c1", tool="run_command", target="pytest", task_key="t1"),
    ev.ToolCallFinished(call_id="c1", tool="run_command", outcome="ok", duration_ms=1.5),
    ev.ApprovalRequested(
        tool_call_id="c2",
        tool="run_command",
        command="curl x",
        directory=".",
        category="network",
        reason="talks to other hosts",
    ),
    ev.ApprovalResolved(tool_call_id="c2", decision="reject", by="policy", message="no"),
    ev.PlanProposed(plan={"project_name": "p"}),
    ev.PlanReviewed(approved=False, by="user", feedback="smaller"),
    ev.ProjectStarted(project_id="p1", resumed=True, recovered=2),
    ev.TaskStarted(task_key="t1", title="Build"),
    ev.TaskFinished(task_key="t1", title="Build", status="failed", detail="judge said no"),
    ev.Progress(completed=1, total=3, pending=2),
    ev.ProjectFinished(project_id="p1", status="completed", counts={"completed": 3}),
    ev.FilesChanged(label="/ask q", files=["a.py"], count=1),
]


def test_every_event_round_trips_through_json() -> None:
    for event in ALL_EVENTS:
        data = json.loads(json.dumps(event.model_dump(mode="json")))
        assert EVENT_ADAPTER.validate_python(data) == event


def test_subscribe_emit_unsubscribe_and_failing_subscriber() -> None:
    def broken(event: ev.Event) -> None:
        raise RuntimeError("boom")

    off = bus.subscribe(broken)  # a failing front end never breaks the run
    try:
        with collecting() as seen:
            notice("one")
        assert [e.text for e in seen if isinstance(e, ev.Notice)] == ["one"]
    finally:
        off()
    with collecting() as seen:
        pass
    notice("not seen")
    assert seen == []


async def test_run_scope_tags_events_also_from_worker_threads() -> None:
    with collecting() as seen:
        with run_scope("run1"):
            emit(ev.Notice(text="loop"))
            await asyncio.to_thread(notice, "thread")  # sync tools run in worker threads
        notice("outside")
    assert [(e.text, e.run_id) for e in seen if isinstance(e, ev.Notice)] == [
        ("loop", "run1"), ("thread", "run1"), ("outside", None)
    ]  # fmt: skip


def test_bus_is_thread_safe() -> None:
    with collecting() as seen:
        threads = [threading.Thread(target=lambda: [notice("x") for _ in range(200)]) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(seen) == 1600


def _render(*events: ev.Event) -> str:
    console = Console(record=True, width=200, color_system=None)
    renderer = ConsoleRenderer(console)
    for e in events:
        renderer(e)
    return console.export_text()


def test_renderer_shows_every_event_without_crashing() -> None:
    out = _render(*ALL_EVENTS)
    assert "hi" in out and "answer" in out and "(answer from the semantic cache)" in out
    assert "→ run_command pytest" in out and "▶ Starting: [t1] Build" in out
    assert "❌ Failed:    [t1] Build: judge said no" in out and "All 3 tasks completed" in out
    assert "Resuming unfinished project p1 (2 crashed task(s) recovered)" in out
    assert "Changed 1 file(s)" in out and "Re-planning" in out


def test_renderer_never_interprets_markup_from_the_model() -> None:
    out = _render(ev.AssistantMessage(text="[bold red]x[/bold red] [link=http://evil]y[/link]"),
                  ev.Notice(text="[red]notice[/red]"))  # fmt: skip
    assert "[bold red]x[/bold red]" in out and "[link=http://evil]y[/link]" in out and "[red]notice[/red]" in out
