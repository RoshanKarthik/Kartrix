"""Per-user locations outside any workspace (a cloned repo can't plant files here).

Standard per-OS folders via ``platformdirs`` (e.g. ``%LOCALAPPDATA%\\kartrix`` on Windows,
``~/Library/Application Support/kartrix`` on macOS, ``~/.local/share/kartrix`` on Linux).
``KARTRIX_HOME`` puts everything under one folder instead (tests, portable installs).
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

_APP = "kartrix"


def _home() -> Path | None:
    value = os.environ.get("KARTRIX_HOME")
    return Path(value) if value else None


def user_data_dir() -> Path:
    """Settings and state that must survive (e.g. trusted skills, connected MCP servers)."""
    home = _home()
    path = home / "data" if home else Path(platformdirs.user_data_dir(_APP, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_cache_dir() -> Path:
    """Things that can be downloaded again (e.g. MCP server binaries)."""
    home = _home()
    path = home / "cache" if home else Path(platformdirs.user_cache_dir(_APP, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path
