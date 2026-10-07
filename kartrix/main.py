"""The interactive REPL: one front end of :mod:`kartrix.core` (renders its events, asks the user)."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.table import Table

from kartrix.config import settings
from kartrix.core.events import bus
from kartrix.core.session import CoreSession
from kartrix.mcp.mcp_client import McpError, McpManager
from kartrix.memory.session import InvalidSessionIdError, new_session, switch_session
from kartrix.observability.logger import get_logger
from kartrix.security import audit, checkpoints, external_tools
from kartrix.security.budget import Budget
from kartrix.security.permissions import MODES, clear_session_allowances, get_mode, set_mode
from kartrix.skills.registry import SkillNotFoundError
from kartrix.skills.skill_tools import get_registry
from kartrix.ui.approval_prompt import ConsoleApprover
from kartrix.ui.console import ConsoleRenderer
from kartrix.ui.plan_review import make_reviewer
from kartrix.ui.views import show_index, show_task_status

console = Console()
logger = get_logger(__name__)
approver = ConsoleApprover(console)


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


async def handle_mcp_command(command: str, arg: str, mcp: McpManager, session_id: str) -> bool:
    """/mcp, /connect <server>, /disconnect <server>. True if the agent's tools changed."""
    if command == "/mcp" or not arg:
        try:
            servers = McpManager.available()
        except McpError as e:
            console.print(f"[red]{e}[/red]")
            return False
        table = Table(title="MCP servers")
        for col in ("server", "status", "tools", "description"):
            table.add_column(col)
        for name, server in servers.items():
            connected = name in mcp.connected
            count = sum(1 for t in mcp.tools if (ext := external_tools.get(t.name)) and ext.server == name)
            status = "[green]connected[/green]" if connected else "off"
            table.add_row(name, status, str(count) if connected else "", server.description)
        console.print(table)
        console.print("[dim]/connect <server> to turn one on; it stays on for later sessions until /disconnect.[/dim]")
        return False
    if command == "/connect":
        console.print(f"[dim]Connecting {arg} (on first use the server is downloaded and verified)...[/dim]")
        try:
            tools = await mcp.connect(arg)
        except McpError as e:
            console.print(f"[red]{e}[/red]")
            await audit.record(actor="user", action="mcp.connect", target=arg, outcome="error", session_id=session_id)
            return False
        approve = sum(1 for t in tools if (ext := external_tools.get(t.name)) and ext.needs_approval)
        console.print(f"[green]✓ {arg} connected: {len(tools)} tools ({approve} need your approval to run).[/green]")
        await audit.record(
            actor="user", action="mcp.connect", target=arg, outcome="ok", session_id=session_id,
            details={"tools": [t.name for t in tools]},
        )  # fmt: skip
        return True
    await mcp.disconnect(arg)
    console.print(f"[dim]{arg} disconnected.[/dim]")
    await audit.record(actor="user", action="mcp.disconnect", target=arg, outcome="ok", session_id=session_id)
    return True


async def handle_skills_command(arg: str, session_id: str) -> bool:
    """/skills, /skills trust <name>, /skills untrust <name>. True if the trusted set changed."""
    registry = get_registry()
    action, _, name = arg.partition(" ")
    if action in ("trust", "untrust") and name.strip():
        name = name.strip()
        try:
            if action == "untrust":
                registry.untrust(name)
                console.print(f"[dim]Skill {name} is no longer approved.[/dim]")
            else:
                skill = registry.trust(name)
                console.print(f"[green]✓ Skill {name} approved[/green] [dim](sha256 {skill.digest[:16]}…)[/dim]")
        except SkillNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return False
        await audit.record(actor="user", action=f"skills.{action}", target=name, outcome="ok", session_id=session_id)
        return True
    if arg:
        console.print("[yellow]Usage: /skills, /skills trust <name>, /skills untrust <name>[/yellow]")
        return False

    skills = registry.skills()
    if not skills:
        console.print("[dim]This repo has no skills (.kartrix/skills/).[/dim]")
        return False
    colors = {"trusted": "green", "untrusted": "yellow", "changed": "red"}
    for skill in skills:
        status = registry.status(skill.name)
        files = [p for p in sorted(skill.skill_dir.rglob("*")) if p.is_file()]
        console.print(
            f"[bold]{skill.name}[/bold] [{colors[status]}]{status}[/{colors[status]}] — {escape(skill.description)}"
        )
        console.print(f"  [dim]{len(files)} files in {skill.skill_dir.relative_to(Path.cwd()).as_posix()}[/dim]")
        if skill.findings:
            rules = ", ".join(sorted({f.rule for f in skill.findings}))
            console.print(f"  [red]⚠ contains text that looks like prompt injection ({rules})[/red]")
    console.print("[dim]Review a skill's files before approving it: /skills trust <name>[/dim]")
    return False


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _list_files(title: str, files: list[str], color: str = "dim", limit: int = 15) -> None:
    if not files:
        return
    console.print(f"[{color}]{title}:[/{color}]")
    for f in files[:limit]:
        console.print(f"  [{color}]{escape(f)}[/{color}]")
    if len(files) > limit:
        console.print(f"  [{color}]… and {len(files) - limit} more[/{color}]")


async def handle_checkpoint_command(command: str, arg: str, session_id: str) -> None:
    """/undo, /redo [force], /checkpoints."""
    store = checkpoints.current()
    if store is None:
        console.print(f"[yellow]Checkpoints are off: {checkpoints.unavailable_reason()}[/yellow]")
        return
    if command == "/checkpoints":
        undo, redo = await asyncio.to_thread(store.entries)
        if not undo and not redo:
            console.print("[dim]No checkpoints yet — every chat turn and /plan task that changes files gets one.[/dim]")
            return
        table = Table(title="Checkpoints (newest first)")
        for col in ("", "when", "change", "files"):
            table.add_column(col)
        for entry in reversed(redo):
            table.add_row("[dim]undone[/dim]", _when(entry.created), escape(entry.label[:70]), str(entry.file_count))
        for i, entry in enumerate(reversed(undo)):
            mark = "[green]/undo →[/green]" if i == 0 else ""
            table.add_row(mark, _when(entry.created), escape(entry.label[:70]), str(entry.file_count))
        console.print(table)
        return

    step = store.undo if command == "/undo" else store.redo
    verb = "undo" if command == "/undo" else "redo"
    result = await asyncio.to_thread(step, arg == "force")
    if result is None:
        console.print(f"[dim]Nothing to {verb}.[/dim]")
        return
    if not result.applied:
        console.print(f"[yellow]These files changed after “{escape(result.entry.label[:80])}”:[/yellow]")
        _list_files("changed since", result.conflicts, "yellow")
        if not sys.stdin.isatty():
            console.print(f"[yellow]Nothing was changed. Run /{verb} force to overwrite them.[/yellow]")
            return
        if not await asyncio.to_thread(Confirm.ask, f"{verb.capitalize()} anyway and overwrite those changes?"):
            console.print("[dim]Nothing was changed.[/dim]")
            return
        result = await asyncio.to_thread(step, True)
        if result is None or not result.applied:
            return
    done = "Undid" if verb == "undo" else "Redid"
    console.print(
        f"[green]✓ {done}: {escape(result.entry.label[:80])}[/green] "
        f"[dim]({len(result.restored)} restored, {len(result.deleted)} deleted)[/dim]"
    )
    _list_files("restored", result.restored)
    _list_files("deleted", result.deleted)
    _list_files("could not delete", result.errors, "red")
    if result.conflicts:
        console.print("[dim]The overwritten versions are kept in the checkpoint store.[/dim]")
    if verb == "undo" and result.entry.label.startswith("task "):
        console.print("[dim]The task stays marked completed in /task_status.[/dim]")
    await audit.record(
        actor="user",
        action=f"checkpoint.{verb}",
        target=result.entry.label,
        outcome="ok",
        session_id=session_id,
        details={
            "entry": result.entry.id,
            "restored": result.restored[:200],
            "deleted": result.deleted[:200],
            "overwritten": result.conflicts[:200],
            "errors": result.errors,
        },
    )


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


async def _run_async():
    logger.info("Starting Kartrix")
    console.print("\n[bold blue]Kartrix[/bold blue] — RAG-powered code assistant")

    unsubscribe = bus.subscribe(ConsoleRenderer(console))
    try:
        core = await CoreSession.start(approver, make_reviewer(console))
    except BaseException:
        unsubscribe()
        raise
    console.print("[green]✓ Ready[/green]\n")
    console.print("Type [bold]'/exit'[/bold] to quit\n")

    try:
        while True:
            user_input = Prompt.ask("[bold green]>[/bold green]")

            if not user_input.strip():
                continue
            if user_input.lower() in ("/exit", "/quit"):
                logger.info("Shutting down")
                console.print("[dim]Goodbye![/dim]")
                break
            elif user_input.startswith("/ask "):
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
            elif user_input == "/mode" or user_input.startswith("/mode "):
                target = user_input.removeprefix("/mode").strip()
                if target:
                    try:
                        previous = get_mode()
                        set_mode(target)
                    except ValueError as e:
                        console.print(f"[red]{e}[/red]")
                    else:
                        await audit.record(
                            actor="user",
                            action="permissions.mode",
                            target=target,
                            outcome="ok",
                            session_id=core.session_id,
                            details={"previous": previous},
                        )
                console.print(f"[dim]Permission mode: {get_mode()} (available: {', '.join(MODES)})[/dim]")
            elif user_input.split()[0] in ("/mcp", "/connect", "/disconnect"):
                command, _, arg = user_input.partition(" ")
                if core.mcp is not None and await handle_mcp_command(command, arg.strip(), core.mcp, core.session_id):
                    core.rebuild_agent()
            elif user_input.split()[0] in ("/undo", "/redo", "/checkpoints"):
                command, _, arg = user_input.partition(" ")
                await handle_checkpoint_command(command, arg.strip(), core.session_id)
            elif user_input == "/budget":
                show_budget(core.last_budget)
            elif user_input == "/skills" or user_input.startswith("/skills "):
                if await handle_skills_command(user_input.removeprefix("/skills").strip(), core.session_id):
                    core.rebuild_agent()
            else:
                logger.warning(f"Unknown command received: {user_input}")
                console.print("[yellow]Unknown command. Try:[/yellow]")
                console.print("  [bold]/ask <question>[/bold]          — ask a question about the codebase")
                console.print("  [bold]/show_index[/bold]              — show all chunks in the index")
                console.print("  [bold]/reindex[/bold]                 — manually re-index current directory")
                console.print("  [bold]/new_session[/bold]             — start a fresh conversation")
                console.print("  [bold]/switch <session_id>[/bold]     — resume a past session")
                console.print("  [bold]/session[/bold]                 — show current session id")
                console.print("  [bold]/plan <goal>[/bold]             — generate and execute a plan")
                console.print("  [bold]/task_status[/bold]             — show task progress for active project")
                console.print("  [bold]/mode \\[read_only|default|auto][/bold] — show or change the permission mode")
                console.print("  [bold]/audit \\[n][/bold]                — show this session's last n tool calls")
                console.print("  [bold]/mcp[/bold]                     — list MCP servers")
                console.print("  [bold]/connect <server>[/bold]        — connect an MCP server (e.g. github)")
                console.print("  [bold]/disconnect <server>[/bold]     — disconnect it")
                console.print("  [bold]/skills \\[trust|untrust <name>][/bold] — review this repo's skills")
                console.print(
                    "  [bold]/undo \\[force][/bold]            — revert the files changed by the last turn/task"
                )
                console.print("  [bold]/redo \\[force][/bold]            — re-apply what /undo reverted")
                console.print("  [bold]/checkpoints[/bold]             — list undo points")
                console.print("  [bold]/budget[/bold]                  — budget limits and the last run's usage")
                console.print(
                    "  [dim]Ctrl+C stops a running turn or plan (twice quits); `kartrix stop` stops all runs[/dim]"
                )
    finally:
        await core.close()
        unsubscribe()


def run():
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
