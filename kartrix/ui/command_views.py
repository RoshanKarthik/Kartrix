"""Terminal front end of the workspace commands (:mod:`kartrix.core.commands`): one flow for the local REPL and
the remote one — parse the line, carry it out through ``call`` (in this process, or on the server), ask before
``/undo`` / ``/redo`` overwrite later changes, render the result."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm
from rich.table import Table

from kartrix.core.commands import CommandError, parse

Call = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


async def execute(console: Console, line: str, call: Call, *, interactive: bool | None = None) -> str | None:
    """Run one command line; the command's name when it changed something (None: nothing changed or it failed).
    ``call`` raises :class:`CommandError` (or returns normally) — other errors are the caller's."""
    try:
        name, args = parse(line)
        if name == "mcp.connect":
            server = escape(args["server"])
            console.print(f"[dim]Connecting {server} (on first use the server is downloaded and verified)...[/dim]")
        result = await call(name, args)
        if name in ("undo", "redo") and _blocked(result):
            _list_files(
                console, "changed since", result["result"]["conflicts"], "yellow", title=_conflict_title(result)
            )
            verb = result["verb"]
            ask = sys.stdin.isatty() if interactive is None else interactive
            if not ask:
                console.print(f"[yellow]Nothing was changed. Run /{verb} force to overwrite them.[/yellow]")
                return None
            question = f"{verb.capitalize()} anyway and overwrite those changes?"
            if not await asyncio.to_thread(Confirm.ask, question, console=console):
                console.print("[dim]Nothing was changed.[/dim]")
                return None
            result = await call(name, {**args, "force": True})
            if _blocked(result):
                return None
    except CommandError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        return None
    render(console, name, result)
    return name if _changed(name, result) else None


def _blocked(result: dict[str, Any]) -> bool:
    step = result.get("result")
    return bool(result.get("available") and step is not None and not step["applied"])


def _changed(name: str, result: dict[str, Any]) -> bool:
    if name in ("undo", "redo"):
        return bool(result.get("result") and result["result"]["applied"])
    if name == "mode":
        return bool(result.get("changed"))
    return name in ("memory.add", "memory.forget", "skills.trust", "skills.untrust", "mcp.connect", "mcp.disconnect")


def _conflict_title(result: dict[str, Any]) -> str:
    return f"These files changed after “{escape(result['result']['label'][:80])}”"


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _list_files(
    console: Console, label: str, files: list[str], color: str = "dim", limit: int = 15, title: str | None = None
) -> None:
    if not files:
        return
    if title:
        console.print(f"[{color}]{title}:[/{color}]")
    else:
        console.print(f"[{color}]{label}:[/{color}]")
    for f in files[:limit]:
        console.print(f"  [{color}]{escape(f)}[/{color}]")
    if len(files) > limit:
        console.print(f"  [{color}]… and {len(files) - limit} more[/{color}]")


def render(console: Console, name: str, result: dict[str, Any]) -> None:
    _RENDERERS[name](console, result)


def _render_checkpoints(console: Console, result: dict[str, Any]) -> None:
    if not result["available"]:
        console.print(f"[yellow]Checkpoints are off: {escape(str(result.get('reason')))}[/yellow]")
        return
    undo, redo = result["undo"], result["redo"]
    if not undo and not redo:
        console.print("[dim]No checkpoints yet — every chat turn and /plan task that changes files gets one.[/dim]")
        return
    table = Table(title="Checkpoints (newest first)")
    for col in ("", "when", "change", "files"):
        table.add_column(col)
    for entry in reversed(redo):
        table.add_row(
            "[dim]undone[/dim]", _when(entry["created"]), escape(entry["label"][:70]), str(entry["file_count"])
        )
    for i, entry in enumerate(reversed(undo)):
        mark = "[green]/undo →[/green]" if i == 0 else ""
        table.add_row(mark, _when(entry["created"]), escape(entry["label"][:70]), str(entry["file_count"]))
    console.print(table)


def _render_step(console: Console, result: dict[str, Any]) -> None:
    verb = result["verb"]
    if not result["available"]:
        console.print(f"[yellow]Checkpoints are off: {escape(str(result.get('reason')))}[/yellow]")
        return
    step = result["result"]
    if step is None:
        console.print(f"[dim]Nothing to {verb}.[/dim]")
        return
    done = "Undid" if verb == "undo" else "Redid"
    console.print(
        f"[green]✓ {done}: {escape(step['label'][:80])}[/green] "
        f"[dim]({len(step['restored'])} restored, {len(step['deleted'])} deleted)[/dim]"
    )
    _list_files(console, "restored", step["restored"])
    _list_files(console, "deleted", step["deleted"])
    _list_files(console, "could not delete", step["errors"], "red")
    if step["conflicts"]:
        console.print("[dim]The overwritten versions are kept in the checkpoint store.[/dim]")
    if verb == "undo" and step["label"].startswith("task "):
        console.print("[dim]The task stays marked completed in /task_status.[/dim]")


def _render_mode(console: Console, result: dict[str, Any]) -> None:
    console.print(f"[dim]Permission mode: {result['mode']} (available: {', '.join(result['modes'])})[/dim]")


def _render_memory_list(console: Console, result: dict[str, Any]) -> None:
    memories = result["memories"]
    if not memories:
        console.print(
            "[dim]No memories yet. /memory add <text> (or --user for a preference); "
            f"{result['project_file']} at the repo root is always loaded.[/dim]"
        )
        return
    table = Table(title="Long-term memory (newest first)")
    for col in ("id", "kind", "scope", "source", "updated", "uses", "content"):
        table.add_column(col)
    for m in memories:
        updated = datetime.fromisoformat(m["updated_at"]).astimezone().strftime("%m-%d %H:%M")
        table.add_row(
            m["id"][:8], m["kind"], m["scope"], m["source"], updated, str(m["uses"]), escape(m["content"][:100])
        )
    console.print(table)
    console.print("[dim]/memory forget <id> removes one.[/dim]")


def _render_memory_add(console: Console, result: dict[str, Any]) -> None:
    where = "every repository" if result["scope"] == "user" else "this repository"
    console.print(f"[green]✓ Remembered ({result['kind']}, {where}): {result['id'][:8]}[/green]")


def _render_memory_forget(console: Console, result: dict[str, Any]) -> None:
    console.print(f"[dim]Forgot: {escape(result['content'][:80])}[/dim]")


def _render_skills_list(console: Console, result: dict[str, Any]) -> None:
    skills = result["skills"]
    if not skills:
        console.print("[dim]This repo has no skills (.kartrix/skills/).[/dim]")
        return
    colors = {"trusted": "green", "untrusted": "yellow", "changed": "red"}
    for skill in skills:
        color = colors.get(skill["status"], "white")
        console.print(
            f"[bold]{escape(skill['name'])}[/bold] [{color}]{skill['status']}[/{color}] — "
            f"{escape(skill['description'])}"
        )
        console.print(f"  [dim]{skill['files']} files in {escape(skill['folder'])}[/dim]")
        if skill["injection_rules"]:
            rules = ", ".join(skill["injection_rules"])
            console.print(f"  [red]⚠ contains text that looks like prompt injection ({escape(rules)})[/red]")
    console.print("[dim]Review a skill's files before approving it: /skills trust <name>[/dim]")


def _render_skills_trust(console: Console, result: dict[str, Any]) -> None:
    name = escape(result["name"])
    if result["trusted"]:
        console.print(f"[green]✓ Skill {name} approved[/green] [dim](sha256 {result['digest'][:16]}…)[/dim]")
    else:
        console.print(f"[dim]Skill {name} is no longer approved.[/dim]")


def _render_mcp_list(console: Console, result: dict[str, Any]) -> None:
    table = Table(title="MCP servers")
    for col in ("server", "status", "tools", "description"):
        table.add_column(col)
    for server in result["servers"]:
        status = "[green]connected[/green]" if server["connected"] else "off"
        tools = str(server["tools"]) if server["connected"] else ""
        table.add_row(escape(server["name"]), status, tools, escape(server["description"]))
    console.print(table)
    console.print("[dim]/connect <server> to turn one on; it stays on for later sessions until /disconnect.[/dim]")


def _render_mcp_connect(console: Console, result: dict[str, Any]) -> None:
    console.print(
        f"[green]✓ {escape(result['server'])} connected: {result['tools']} tools "
        f"({result['need_approval']} need your approval to run).[/green]"
    )


def _render_mcp_disconnect(console: Console, result: dict[str, Any]) -> None:
    console.print(f"[dim]{escape(result['server'])} disconnected.[/dim]")


_RENDERERS: dict[str, Callable[[Console, dict[str, Any]], None]] = {
    "checkpoints": _render_checkpoints,
    "undo": _render_step,
    "redo": _render_step,
    "mode": _render_mode,
    "memory.list": _render_memory_list,
    "memory.add": _render_memory_add,
    "memory.forget": _render_memory_forget,
    "skills.list": _render_skills_list,
    "skills.trust": _render_skills_trust,
    "skills.untrust": _render_skills_trust,
    "mcp.list": _render_mcp_list,
    "mcp.connect": _render_mcp_connect,
    "mcp.disconnect": _render_mcp_disconnect,
}
