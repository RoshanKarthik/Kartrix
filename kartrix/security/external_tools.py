"""Registry of tools whose results come from outside the workspace (MCP servers).

Filled when MCP tools are loaded (``kartrix.mcp.mcp_client``); read by the content guard
(mark + taint) and the approval middleware (tools that may change external data).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalTool:
    server: str
    tool: str  # the name the agent sees
    needs_approval: bool  # may change data outside the workspace
    reason: str  # shown to the user when asking


_tools: dict[str, ExternalTool] = {}


def register(tool: ExternalTool) -> None:
    _tools[tool.tool] = tool


def get(name: str) -> ExternalTool | None:
    return _tools.get(name)


def unregister(name: str) -> None:
    _tools.pop(name, None)


def clear() -> None:
    _tools.clear()
