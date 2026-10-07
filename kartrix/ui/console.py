"""The REPL's view of the event stream: renders core events with Rich.

Questions are not events — approvals go through :class:`~kartrix.ui.approval_prompt.ConsoleApprover`
and plan reviews through :func:`kartrix.ui.plan_review.review_plan`. Text from the model, tools or
files is printed without interpreting Rich markup.
"""

from __future__ import annotations

import threading

from rich.console import Console
from rich.markup import escape

from kartrix.core import events as ev

_NOTICE_STYLE = {"info": "dim", "success": "green", "warning": "yellow", "error": "red"}
_NOTICE_PREFIX = {"success": "✓ ", "warning": "", "error": "", "info": ""}


class ConsoleRenderer:
    def __init__(self, console: Console | None = None, *, show_tool_calls: bool = True) -> None:
        self.console = console or Console()
        self.show_tool_calls = show_tool_calls
        self._held: list[ev.Event] | None = None
        self._lock = threading.Lock()
        self._streamed: set[str | None] = set()  # runs whose answer was already printed token by token

    def hold(self) -> None:
        """Keep events back (e.g. while the prompt waits for input) until :meth:`release`."""
        with self._lock:
            if self._held is None:
                self._held = []

    def release(self) -> None:
        """Show the events held back, then render live again."""
        with self._lock:
            held, self._held = self._held or [], None
        for event in held:
            self._render(event)

    def _line(self, text: str, style: str = "") -> None:
        self.console.print(f"[{style}]{escape(text)}[/{style}]" if style else escape(text))

    def __call__(self, event: ev.Event) -> None:
        with self._lock:
            if self._held is not None:
                self._held.append(event)
                return
        self._render(event)

    def _render(self, event: ev.Event) -> None:
        match event:
            case ev.Notice(level=level, text=text):
                self._line(_NOTICE_PREFIX[level] + text, _NOTICE_STYLE[level])
            case ev.RunStarted(kind="ask", input=question):
                self._line(f"Working on: {question[:200]}… (Ctrl+C stops)", "dim")
            case ev.AssistantDelta(text=text, run_id=run_id):
                if run_id not in self._streamed:
                    self._streamed.add(run_id)
                    self.console.print()
                self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
            case ev.AssistantMessage(text=text, cached=cached, run_id=run_id):
                if run_id in self._streamed:
                    self._streamed.discard(run_id)
                    self.console.print()  # the answer is on screen already; end its line
                    return
                if cached:
                    self._line("(answer from the semantic cache)", "dim")
                self.console.print(text, markup=False, highlight=False)
            case ev.RunFinished(status="error", detail=detail):
                self._line(f"Error: {detail}", "red")
            case ev.ToolCallStarted(tool=tool, target=target) if self.show_tool_calls:
                self._line(f"  → {tool} {(target or '')[:120]}".rstrip(), "dim")
            case ev.ToolCallFinished(tool=tool, outcome=outcome) if self.show_tool_calls and outcome != "ok":
                self._line(f"  ✗ {tool}: {outcome}", "yellow" if outcome in ("declined", "stopped") else "red")
            case ev.ContextAssembled(sections=sections, stale=stale) if sections:
                parts = [f"{s.name} {s.tokens}/{s.budget}" + (f" ({s.items})" if s.items else "") for s in sections]
                note = f" · {stale} possibly stale" if stale else ""
                self._line(f"◇ context: {' · '.join(parts)} tokens{note}", "dim")
            case ev.AgentStep(agent="router", status="finished", summary=route):
                self._line(f"◆ router: {route}", "dim")
            case ev.AgentStep(agent=agent, status="started") if agent in ("explorer", "coder", "reviewer"):
                self._line(f"◆ {agent} working…", "cyan")
            case ev.AgentStep(agent="reviewer", status="finished", summary=summary):
                self._line(f"◆ reviewer: {summary}", "green" if summary == "approved" else "yellow")
            case ev.PlanReviewed(approved=False):
                self._line("Re-planning with your feedback...", "dim")
            case ev.ProjectStarted(project_id=pid, resumed=True, recovered=recovered):
                more = f" ({recovered} crashed task(s) recovered)" if recovered else ""
                self._line(f"↩ Resuming unfinished project {pid}{more}", "yellow")
            case ev.ProjectStarted(project_id=pid, resumed=False):
                self._line(f"Project {pid} saved.", "dim")
            case ev.Progress(completed=c, total=t, in_progress=i, pending=p, failed=f):
                self._line(f"Progress: {c}/{t} completed · {i} in progress · {p} pending · {f} failed", "dim")
            case ev.TaskStarted(task_key=key, title=title):
                self.console.print("\n[bold]▶ Starting:[/bold] " + escape(f"[{key}] {title}"))
            case ev.TaskFinished(task_key=key, title=title, status=status, detail=detail):
                self._task_finished(key, title, status, detail)
            case ev.ProjectFinished(status=status, counts=counts):
                self._project_finished(status, counts)
            case ev.FilesChanged(count=count):
                self._line(f"Changed {count} file(s) — /undo reverts them", "dim")
            case _:
                pass

    def _task_finished(self, key: str, title: str, status: str, detail: str | None) -> None:
        name = escape(f"[{key}] {title}")
        if status == "completed":
            self.console.print(f"[green]✅ Completed:[/green] {name}")
        elif status == "stopped":
            self.console.print(f"[yellow]⏸ Stopped:[/yellow]   {name} — back to pending")
        else:
            self.console.print(f"[red]❌ Failed:[/red]    {name}: {escape((detail or '')[:120])}")

    def _project_finished(self, status: str, counts: dict[str, int]) -> None:
        if status == "planned":
            self._line("Plan saved — run /plan again to execute it.", "dim")
            return
        if status == "stopped":
            self._line("⏸ Plan stopped. Run /plan again to continue where it stopped (with a new budget).", "yellow")
            return
        completed, failed, blocked = counts.get("completed", 0), counts.get("failed", 0), counts.get("blocked", 0)
        if status == "completed":
            self.console.print(f"\n[bold green]🎉 All {completed} tasks completed successfully![/bold green]")
            return
        self.console.print("\n[bold yellow]⚠ Execution finished with issues:[/bold yellow]")
        self.console.print(f"  ✅ Completed: {completed}")
        if failed:
            self.console.print(f"  ❌ Failed:    {failed}  (run /task_status to review)")
        if blocked:
            self.console.print(f"  🚫 Blocked:   {blocked}  (dependencies failed)")
        if skipped := counts.get("skipped", 0):
            self.console.print(f"  ⏭ Skipped:   {skipped}")
