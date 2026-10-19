"""The REPL commands that change a workspace's state — ``/undo``, ``/redo``, ``/checkpoints``, ``/mode``,
``/memory``, ``/skills``, ``/mcp``, ``/connect``, ``/disconnect`` — as data operations.

:func:`parse` turns a command line into a name and arguments, :func:`run` carries it out and returns a JSON-able
result (or raises :class:`CommandError` with a message for the user). The local REPL calls :func:`run` in its own
process; through a server the same name and arguments travel to the workspace's worker, which calls :func:`run`
there (:mod:`kartrix.server.worker`). Both front ends render the result with :mod:`kartrix.ui.command_views`, so
local and remote output are the same. Every change is audited with the session it came from.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # the remote client imports this module: no LangChain at startup
    from kartrix.mcp.mcp_client import McpManager

# name → True when it changes state (refused while a run is in progress in the workspace)
COMMANDS: dict[str, bool] = {
    "checkpoints": False,
    "undo": True,
    "redo": True,
    "mode": True,  # also lists the mode (no "target")
    "memory.list": False,
    "memory.add": True,
    "memory.forget": True,
    "skills.list": False,
    "skills.trust": True,
    "skills.untrust": True,
    "mcp.list": False,
    "mcp.connect": True,
    "mcp.disconnect": True,
}
SLASH_COMMANDS = ("/undo", "/redo", "/checkpoints", "/mode", "/memory", "/skills", "/mcp", "/connect", "/disconnect")
MEMORY_USAGE = "Usage: /memory, /memory add [--user] <text>, /memory forget <id>"
SKILLS_USAGE = "Usage: /skills, /skills trust <name>, /skills untrust <name>"


class CommandError(Exception):
    """A command that can't be carried out; the message is for the user."""


class CommandHost(Protocol):
    """What a command needs from the core (:class:`kartrix.core.session.CoreSession`)."""

    session_id: str
    mcp: McpManager | None

    def rebuild_agent(self) -> None: ...


def is_command(line: str) -> bool:
    return line.split(maxsplit=1)[0] in SLASH_COMMANDS if line.strip() else False


def parse(line: str) -> tuple[str, dict[str, Any]]:
    """``/memory add --user be brief`` → ``("memory.add", {"content": "be brief", "user": True})``."""
    command, _, rest = line.strip().partition(" ")
    arg = rest.strip()
    if command in ("/undo", "/redo"):
        if arg not in ("", "force"):
            raise CommandError(f"Usage: {command} [force]")
        return command[1:], {"force": arg == "force"}
    if command == "/checkpoints":
        return "checkpoints", {}
    if command == "/mode":
        return "mode", {"target": arg or None}
    if command == "/memory":
        action, _, value = arg.partition(" ")
        value = value.strip()
        if not arg:
            return "memory.list", {}
        if action == "add" and value:
            user = value.startswith("--user")
            content = value.removeprefix("--user").strip()
            if content:
                return "memory.add", {"content": content, "user": user}
        if action == "forget" and value:
            return "memory.forget", {"id": value}
        raise CommandError(MEMORY_USAGE)
    if command == "/skills":
        action, _, name = arg.partition(" ")
        if not arg:
            return "skills.list", {}
        if action in ("trust", "untrust") and name.strip():
            return f"skills.{action}", {"name": name.strip()}
        raise CommandError(SKILLS_USAGE)
    if command in ("/connect", "/disconnect") and arg:
        return f"mcp.{command[1:]}", {"server": arg}
    if command in ("/mcp", "/connect", "/disconnect"):
        return "mcp.list", {}
    raise CommandError(f"Unknown command {command}")


async def run(host: CommandHost, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Carry out one command in this process's workspace (the current directory)."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise CommandError(f"Unknown command {name!r}")
    return await handler(host, args)


async def _record(host: CommandHost, action: str, target: str, details: dict[str, Any] | None = None) -> None:
    from kartrix.security import audit

    await audit.record(
        actor="user", action=action, target=target, outcome="ok", session_id=host.session_id, details=details
    )


# ── checkpoints ───────────────────────────────────────────────────────


def _entry(entry: Any) -> dict[str, Any]:
    return {"id": entry.id, "label": entry.label, "created": entry.created, "file_count": entry.file_count}


async def _checkpoints(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.security import checkpoints

    store = checkpoints.current()
    if store is None:
        return {"available": False, "reason": checkpoints.unavailable_reason()}
    undo, redo = await asyncio.to_thread(store.entries)
    return {"available": True, "undo": [_entry(e) for e in undo], "redo": [_entry(e) for e in redo]}


async def _step(host: CommandHost, verb: str, force: bool) -> dict[str, Any]:
    from kartrix.security import checkpoints

    store = checkpoints.current()
    if store is None:
        return {"available": False, "reason": checkpoints.unavailable_reason(), "verb": verb}
    step = store.undo if verb == "undo" else store.redo
    result = await asyncio.to_thread(step, force)
    if result is None:
        return {"available": True, "verb": verb, "result": None}
    out = {
        "label": result.entry.label,
        "applied": result.applied,
        "restored": result.restored,
        "deleted": result.deleted,
        "conflicts": result.conflicts,
        "errors": result.errors,
    }
    if result.applied:
        await _record(
            host,
            f"checkpoint.{verb}",
            result.entry.label,
            {
                "entry": result.entry.id,
                "restored": result.restored[:200],
                "deleted": result.deleted[:200],
                "overwritten": result.conflicts[:200],
                "errors": result.errors,
            },
        )
    return {"available": True, "verb": verb, "result": out}


async def _undo(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    return await _step(host, "undo", bool(args.get("force")))


async def _redo(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    return await _step(host, "redo", bool(args.get("force")))


# ── permission mode ───────────────────────────────────────────────────


async def _mode(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.security.permissions import MODES, get_mode, set_mode

    target = args.get("target")
    changed = False
    if target:
        previous = get_mode()
        try:
            set_mode(str(target))
        except ValueError as e:
            raise CommandError(str(e)) from None
        changed = True
        await _record(host, "permissions.mode", str(target), {"previous": previous})
    return {"mode": get_mode(), "modes": list(MODES), "changed": changed}


# ── long-term memory ──────────────────────────────────────────────────


async def _memory_list(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.memory import long_term

    memories = await long_term.list_memories()
    return {
        "project_file": long_term.PROJECT_FILE,
        "memories": [
            {
                "id": str(m.id),
                "kind": m.kind,
                "scope": m.scope,
                "source": m.source,
                "updated_at": m.updated_at.isoformat(),
                "uses": m.uses,
                "content": m.content,
            }
            for m in memories
        ],
    }


async def _memory_add(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.memory import long_term

    content = str(args.get("content") or "").strip()
    if not content:
        raise CommandError(MEMORY_USAGE)
    kind: long_term.Kind = "preference" if args.get("user") else "fact"
    memory_id = await long_term.remember(content, kind, source="user")
    await _record(host, "memory.add", memory_id)
    return {"id": memory_id, "kind": kind, "scope": "user" if args.get("user") else "project"}


async def _memory_forget(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.memory import long_term

    prefix = str(args.get("id") or "").strip().lower()
    if not prefix:
        raise CommandError(MEMORY_USAGE)
    matches = [m for m in await long_term.list_memories() if str(m.id).startswith(prefix)]
    if len(matches) != 1:
        raise CommandError("No memory with that id." if not matches else "Ambiguous id prefix.")
    await long_term.forget(str(matches[0].id))
    await _record(host, "memory.forget", str(matches[0].id))
    return {"id": str(matches[0].id), "content": matches[0].content}


# ── skills ────────────────────────────────────────────────────────────


async def _skills_list(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.skills.skill_tools import get_registry

    registry = get_registry()
    skills = []
    for skill in registry.skills():
        files = [p for p in sorted(skill.skill_dir.rglob("*")) if p.is_file()]
        try:
            folder = skill.skill_dir.relative_to(Path.cwd()).as_posix()
        except ValueError:
            folder = skill.skill_dir.as_posix()
        skills.append(
            {
                "name": skill.name,
                "status": registry.status(skill.name),
                "description": skill.description,
                "files": len(files),
                "folder": folder,
                "injection_rules": sorted({f.rule for f in skill.findings}),
            }
        )
    return {"skills": skills}


async def _skills_trust(host: CommandHost, args: dict[str, Any], trust: bool = True) -> dict[str, Any]:
    from kartrix.skills.registry import SkillNotFoundError
    from kartrix.skills.skill_tools import get_registry

    name = str(args.get("name") or "").strip()
    if not name:
        raise CommandError(SKILLS_USAGE)
    registry = get_registry()
    digest = None
    try:
        if trust:
            digest = registry.trust(name).digest
        else:
            registry.untrust(name)
    except SkillNotFoundError as e:
        raise CommandError(str(e)) from None
    action = "trust" if trust else "untrust"
    await _record(host, f"skills.{action}", name)
    host.rebuild_agent()
    return {"name": name, "trusted": trust, "digest": digest}


async def _skills_untrust(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    return await _skills_trust(host, args, trust=False)


# ── MCP servers ───────────────────────────────────────────────────────


def _mcp(host: CommandHost) -> McpManager:
    if host.mcp is None:
        raise CommandError("MCP servers are off here (a hosted server doesn't run them).")
    return host.mcp


async def _mcp_list(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.mcp.mcp_client import McpError, McpManager
    from kartrix.security import external_tools

    mcp = _mcp(host)
    try:
        servers = McpManager.available()
    except McpError as e:
        raise CommandError(str(e)) from None
    listed = []
    for name, server in servers.items():
        connected = name in mcp.connected
        count = sum(1 for t in mcp.tools if (ext := external_tools.get(t.name)) and ext.server == name)
        listed.append(
            {
                "name": name,
                "connected": connected,
                "tools": count if connected else 0,
                "description": server.description,
            }
        )
    return {"servers": listed}


async def _mcp_connect(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    from kartrix.mcp.mcp_client import McpError
    from kartrix.security import audit, external_tools

    mcp = _mcp(host)
    name = str(args.get("server") or "").strip()
    try:
        tools = await mcp.connect(name)
    except McpError as e:
        await audit.record(actor="user", action="mcp.connect", target=name, outcome="error", session_id=host.session_id)
        raise CommandError(str(e)) from None
    need_approval = sum(1 for t in tools if (ext := external_tools.get(t.name)) and ext.needs_approval)
    await _record(host, "mcp.connect", name, {"tools": [t.name for t in tools]})
    host.rebuild_agent()
    return {"server": name, "tools": len(tools), "need_approval": need_approval}


async def _mcp_disconnect(host: CommandHost, args: dict[str, Any]) -> dict[str, Any]:
    mcp = _mcp(host)
    name = str(args.get("server") or "").strip()
    await mcp.disconnect(name)
    await _record(host, "mcp.disconnect", name)
    host.rebuild_agent()
    return {"server": name}


_HANDLERS: dict[str, Callable[[CommandHost, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "checkpoints": _checkpoints,
    "undo": _undo,
    "redo": _redo,
    "mode": _mode,
    "memory.list": _memory_list,
    "memory.add": _memory_add,
    "memory.forget": _memory_forget,
    "skills.list": _skills_list,
    "skills.trust": _skills_trust,
    "skills.untrust": _skills_untrust,
    "mcp.list": _mcp_list,
    "mcp.connect": _mcp_connect,
    "mcp.disconnect": _mcp_disconnect,
}
