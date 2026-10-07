"""Connects MCP servers on request and vets their tools (B7).

- **Opt-in:** no server starts until the user runs ``/connect <name>``; the choice is remembered in
  the per-user data folder (not the workspace, which a cloned repo controls).
- **One persistent session per server**, kept open until ``/disconnect`` or exit — the GitHub
  server keeps its browser-login token in memory, so a new process per call would mean a new login.
  Sessions must be closed from the task that opened them (the REPL's main task).
- **Vetting:** only tools on the server's allow-list are exposed; tools whose names collide with
  Kartrix's own are dropped (a server can't shadow ``read_file`` or ``run_command``); tools whose
  description or schema looks like prompt injection are dropped ("tool poisoning"); descriptions
  are cleaned and length-capped. Every tool is registered as external (results marked untrusted)
  and needs approval unless the server marks it read-only (``readOnlyHint``) and it isn't listed
  under ``tools.approve``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from kartrix.config import settings
from kartrix.mcp.binaries import BinaryError, ensure_binary
from kartrix.mcp.mcp_config import McpConfigError, McpServer, load_mcp_configs
from kartrix.observability.logger import get_logger
from kartrix.paths import user_cache_dir, user_data_dir
from kartrix.security import external_tools
from kartrix.security.external_tools import ExternalTool
from kartrix.security.injection import scan, strip_hidden

logger = get_logger(__name__)

_CONNECT_TIMEOUT = 120.0  # first start may include the download
_DESCRIPTION_CHARS = 2000
_STATE_FILE = "mcp_servers.json"


class McpError(Exception):
    """Safe to show to the user."""


# ── remembered opt-in ─────────────────────────────────────────────────


def enabled_servers() -> list[str]:
    try:
        data = json.loads((user_data_dir() / _STATE_FILE).read_text(encoding="utf-8"))
        return [s for s in data.get("connected", []) if isinstance(s, str)]
    except (OSError, ValueError, AttributeError):
        return []


def _remember(name: str, connected: bool) -> None:
    names = [s for s in enabled_servers() if s != name] + ([name] if connected else [])
    path = user_data_dir() / _STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"connected": names}, indent=2), encoding="utf-8")
    tmp.replace(path)


def _server_log(name: str) -> Path:
    folder = Path(settings.logging.file).parent if settings.logging.file else user_cache_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"mcp-{name}.log"


# ── vetting ───────────────────────────────────────────────────────────


def vet_tools(server_name: str, server: McpServer, tools: Iterable[BaseTool], reserved: set[str]) -> list[BaseTool]:
    """The tools the agent may use, each registered in ``external_tools``."""
    allow = server.tools.allow
    kept: list[BaseTool] = []
    for tool in tools:
        name = tool.name
        if allow != "*" and name not in allow:
            continue
        if name in reserved or external_tools.get(name) is not None:
            logger.warning(
                "MCP tool name collides with an existing tool; dropped", extra={"server": server_name, "tool": name}
            )
            continue
        schema = tool.args_schema if isinstance(tool.args_schema, dict) else {}
        findings = scan(f"{tool.description or ''}\n{json.dumps(schema, ensure_ascii=False)}")
        if findings:
            logger.warning(
                "MCP tool description looks like prompt injection; dropped",
                extra={"server": server_name, "tool": name, "rules": [f.rule for f in findings]},
            )
            continue
        description, _ = strip_hidden(tool.description or "")
        tool.description = description[:_DESCRIPTION_CHARS]
        metadata: dict[str, Any] = tool.metadata or {}
        read_only = metadata.get("readOnlyHint") is True
        needs_approval = name in server.tools.approve or not read_only
        reason = f"{'deletes or overwrites' if metadata.get('destructiveHint') else 'may change'} data on {server_name}"
        external_tools.register(ExternalTool(server_name, name, needs_approval, reason))
        kept.append(tool)
    return kept


# ── sessions ──────────────────────────────────────────────────────────


class McpManager:
    """Owns the MCP sessions of the running Kartrix process."""

    def __init__(self, reserved_names: Iterable[str]) -> None:
        self._reserved = set(reserved_names)
        self._sessions: dict[str, AsyncExitStack] = {}
        self._tools: dict[str, list[BaseTool]] = {}

    @property
    def tools(self) -> list[BaseTool]:
        return [t for tools in self._tools.values() for t in tools]

    @property
    def connected(self) -> list[str]:
        return list(self._tools)

    @staticmethod
    def available() -> dict[str, McpServer]:
        try:
            return load_mcp_configs()
        except McpConfigError as e:
            raise McpError(str(e)) from None

    async def connect(self, name: str, *, remember: bool = True) -> list[BaseTool]:
        if name in self._tools:
            return self._tools[name]
        servers = self.available()
        server = servers.get(name)
        if server is None:
            raise McpError(f"unknown MCP server {name!r} (configured: {', '.join(servers) or 'none'})")
        try:
            command = (
                str(await asyncio.to_thread(ensure_binary, server.binary.spec())) if server.binary else server.command
            )
        except BinaryError as e:
            raise McpError(str(e)) from None
        assert command is not None  # noqa: S101 — exactly one of binary/command is set (validated)
        if server.allow_unpinned:
            logger.warning("Starting an unpinned MCP server", extra={"server": name, "command": command})

        # stdio_client adds only a safe subset of Kartrix's environment (PATH, HOME, …) to `env`.
        params = StdioServerParameters(command=command, args=server.args, env=server.resolved_env())
        stack = AsyncExitStack()
        try:
            # The server's own log lines go to a file, not into the REPL.
            errlog = stack.enter_context(_server_log(name).open("a", encoding="utf-8"))
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), _CONNECT_TIMEOUT)
            raw = await asyncio.wait_for(load_mcp_tools(session, server_name=name), _CONNECT_TIMEOUT)
        except BaseException as e:
            await stack.aclose()
            if isinstance(e, Exception):
                raise McpError(f"could not start the {name} MCP server: {type(e).__name__}: {e}") from None
            raise
        tools = vet_tools(name, server, raw, self._reserved)
        self._sessions[name] = stack
        self._tools[name] = tools
        if remember:
            _remember(name, True)
        logger.info("MCP server connected", extra={"server": name, "tools": len(tools), "offered": len(raw)})
        return tools

    async def disconnect(self, name: str, *, forget: bool = True) -> None:
        stack = self._sessions.pop(name, None)
        for tool in self._tools.pop(name, []):
            external_tools.unregister(tool.name)
        if stack is not None:
            await stack.aclose()
        if forget:
            _remember(name, False)

    async def connect_remembered(self) -> dict[str, str]:
        """Reconnect the servers the user connected before; returns name → error for failures."""
        errors: dict[str, str] = {}
        for name in enabled_servers():
            try:
                await self.connect(name, remember=False)
            except McpError as e:
                errors[name] = str(e)
        return errors

    async def close(self) -> None:
        for name in list(self._sessions):
            await self.disconnect(name, forget=False)
