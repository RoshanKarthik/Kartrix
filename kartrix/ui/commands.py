"""The REPL's slash commands (loaded in the background with the core, so the prompt appears at once)."""

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kartrix.config import settings
from kartrix.core import commands as core_commands
from kartrix.core.session import CoreSession
from kartrix.memory.session import InvalidSessionIdError, new_session, switch_session
from kartrix.observability.logger import get_logger
from kartrix.security import audit
from kartrix.security.budget import Budget
from kartrix.security.permissions import clear_session_allowances
from kartrix.ui.command_views import execute
from kartrix.ui.views import show_index, show_task_status

console = Console()
logger = get_logger(__name__)


async def show_audit(session_id: str, arg: str) -> None:
    """Print the newest audit rows of the current session."""
    limit = int(arg) if arg.isdigit() else 20
    rows = await audit.recent(session_id, limit)
    if not rows:
        console.print("[dim]No audit entries for this session yet.[/dim]")
        return
    table = Table(title=f"Audit log — last {len(rows)} entries")
    for col in ("time", "action", "target", "outcome", "ms"):
        table.add_column(col)
    colors = {"ok": "green", "denied": "red", "error": "red", "needs_approval": "yellow", "declined": "yellow"}
    for row in reversed(rows):
        color = colors.get(row.outcome, "white")
        table.add_row(
            row.ts.astimezone().strftime("%H:%M:%S"),
            row.action,
            (row.target or "")[:60],
            f"[{color}]{row.outcome}[/{color}]",
            str(row.details.get("duration_ms", "")),
        )
    console.print(table)


def show_trace(ref: str | None) -> None:
    """/trace [run|last]: the recent runs, or one run's timeline (kartrix.observability.tracing)."""
    from kartrix.observability import tracing

    if ref is None:
        console.print(tracing.render_list(tracing.list_traces()), markup=False, highlight=False)
        return
    path = tracing.find(ref)
    if path is None:
        console.print(f"[yellow]No trace matches {escape(ref)!r}.[/yellow]")
        return
    console.print(tracing.render(tracing.summarize(tracing.load(path))), markup=False, highlight=False)


def show_budget(last_run: Budget | None) -> None:
    """/budget: the limits and what the last run used."""
    cfg = settings.budgets
    table = Table(title="Budgets (config budgets.*)")
    for col in ("run", "tokens", "cost", "tool calls", "time"):
        table.add_column(col)
    for kind, lim in (("chat turn", cfg.turn), ("/plan", cfg.plan)):
        table.add_row(
            kind,
            f"{lim.max_tokens:,}" if lim.max_tokens else "—",
            f"${lim.max_cost_usd:.2f}" if lim.max_cost_usd else "—",
            str(lim.max_tool_calls or "—"),
            f"{lim.max_seconds:.0f} s" if lim.max_seconds else "—",
        )
    console.print(table)
    if last_run is not None:
        stopped = f" — stopped: {last_run.stop_reason}" if last_run.stop_reason else ""
        console.print(f"[dim]Last run ({last_run.kind}): {last_run.summary()}{stopped}[/dim]")
    console.print("[dim]Ctrl+C stops a run (twice quits); `kartrix stop` in another terminal stops all runs.[/dim]")


def print_help() -> None:
    console.print("Commands:")
    console.print("  [bold]/ask <question>[/bold]          — ask a question about the codebase")
    console.print("  [bold]/show_index[/bold]              — show all chunks in the index")
    console.print("  [bold]/reindex[/bold]                 — manually re-index current directory")
    console.print("  [bold]/new_session[/bold]             — start a fresh conversation")
    console.print("  [bold]/switch <session_id>[/bold]     — resume a past session")
    console.print("  [bold]/session[/bold]                 — show current session id")
    console.print("  [bold]/plan <goal>[/bold]             — generate and execute a plan")
    console.print("  [bold]/task_status[/bold]             — show task progress for active project")
    console.print("  [bold]/mode \\[read_only|default|auto][/bold] — show or change the permission mode")
    console.print("  [bold]/model \\[id][/bold]              — list the models, or switch to one")
    console.print("  [bold]/audit \\[n][/bold]                — show this session's last n tool calls")
    console.print("  [bold]/mcp[/bold]                     — list MCP servers")
    console.print("  [bold]/connect <server>[/bold]        — connect an MCP server (e.g. github)")
    console.print("  [bold]/disconnect <server>[/bold]     — disconnect it")
    console.print("  [bold]/skills \\[trust|untrust <name>][/bold] — review this repo's skills")
    console.print("  [bold]/undo \\[force][/bold]            — revert the files changed by the last turn/task")
    console.print("  [bold]/redo \\[force][/bold]            — re-apply what /undo reverted")
    console.print("  [bold]/checkpoints[/bold]             — list undo points")
    console.print("  [bold]/budget[/bold]                  — budget limits and the last run's usage")
    console.print("  [bold]/trace \\[run|last][/bold]        — recent runs, or one run's timeline")
    console.print("  [bold]/memory \\[add \\[--user] <text>|forget <id>][/bold] — long-term memory")
    console.print("  [dim]Ctrl+C stops a running turn or plan (twice quits); `kartrix stop` stops all runs[/dim]")


async def dispatch(core: CoreSession, user_input: str) -> None:
    """Run one REPL command (anything but /exit and /help, which work before startup finishes). Text that isn't a
    command is a request — the same as ``/ask``."""
    if core_commands.is_command(user_input):
        await execute(console, user_input, lambda name, args: core_commands.run(core, name, args))
    elif user_input.startswith("/ask ") or not user_input.startswith("/"):
        question = user_input.removeprefix("/ask ").strip()
        logger.info(f"Ask command received: {question}")
        await core.ask(question)
    elif user_input == "/reindex":
        console.print("[dim]Re-indexing current directory...[/dim]")
        await core.reindex()
        if core.semantic_cache is not None:
            await core.invalidate_cache()
            console.print("[dim]Semantic cache invalidated for this repo.[/dim]")
        console.print("[green]✓ Re-index complete.[/green]")
    elif user_input == "/new_session":
        clear_session_allowances()
        await core.use_session(new_session())
        console.print(f"[green]New session started: {core.session_id}[/green]")
    elif user_input.startswith("/switch "):
        target = user_input.removeprefix("/switch ").strip()
        try:
            switched = switch_session(target)
        except InvalidSessionIdError:
            console.print("[red]Invalid session id — expected a UUID like the one shown by /session.[/red]")
        else:
            clear_session_allowances()
            await core.use_session(switched)
            console.print(f"[green]Switched to session: {switched}[/green]")
            await core.finish_pending_approval()
    elif user_input == "/session":
        console.print(f"[dim]Current session: {core.session_id}[/dim]")
    elif user_input == "/show_index":
        logger.info("Showing index")
        await show_index(console, Path.cwd())
    elif user_input.startswith("/plan "):
        goal = user_input.removeprefix("/plan ").strip()
        logger.info(f"Plan command received: {goal}")
        await core.plan(goal)
    elif user_input == "/task_status":
        await show_task_status(console)
    elif user_input == "/audit" or user_input.startswith("/audit "):
        await show_audit(core.session_id, user_input.removeprefix("/audit").strip())
    elif user_input == "/model" or user_input.startswith("/model "):
        target = user_input.removeprefix("/model").strip()
        if not target:
            from kartrix.ui.model_list import local_catalog, render_catalog

            render_catalog(console, local_catalog(), "this install", current=core.model_id)
            console.print("[dim]/model <id> switches; the conversation continues with the new model.[/dim]")
            return
        try:
            core.set_model(target)
        except ValueError as e:
            console.print(f"[red]{escape(str(e))}[/red]")
    elif user_input == "/budget":
        show_budget(core.last_budget)
    elif user_input == "/trace" or user_input.startswith("/trace "):
        show_trace(user_input.removeprefix("/trace").strip() or None)
    else:
        logger.warning(f"Unknown command received: {user_input}")
        console.print("[yellow]Unknown command.[/yellow]")
        print_help()
