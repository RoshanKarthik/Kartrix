"""Workspace jail (B1): escapes, protected paths and Windows path tricks are rejected."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kartrix.security import workspace as ws_mod
from kartrix.security.workspace import (
    CASE_INSENSITIVE,
    Workspace,
    WorkspaceError,
    check_windows_path,
    set_workspace,
)


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    root = tmp_path / "ws"
    root.mkdir()
    previous = ws_mod._current
    yield set_workspace(root)
    ws_mod._current = previous


def _link(target: Path, link: Path) -> None:
    """Directory symlink, or a junction on Windows when symlinks need admin rights."""
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        if sys.platform != "win32" or not target.is_dir():
            pytest.skip("symlinks not available")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_relative_and_absolute_paths_inside_resolve(ws: Workspace) -> None:
    assert ws.resolve("src/app.py") == ws.root / "src" / "app.py"
    assert ws.resolve(".") == ws.root
    assert ws.resolve(str(ws.root / "a.txt")) == ws.root / "a.txt"
    assert ws.resolve("src/../b.txt") == ws.root / "b.txt"


@pytest.mark.parametrize("raw", ["..", "../x", "a/../../x", "a/b/../../../x"])
def test_dot_dot_escape_blocked(ws: Workspace, raw: str) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve(raw)


def test_absolute_path_outside_blocked(ws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve(str(tmp_path / "other.txt"))


def test_leading_slash_gets_a_hint(ws: Workspace) -> None:
    with pytest.raises(WorkspaceError, match="drop the leading slash"):
        ws.resolve("/.env")


def test_sibling_with_same_prefix_blocked(ws: Workspace, tmp_path: Path) -> None:
    (tmp_path / "ws2").mkdir()
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve(str(tmp_path / "ws2" / "x.txt"))


@pytest.mark.parametrize("raw", ["", "   ", "a\x00b"])
def test_empty_and_nul_rejected(ws: Workspace, raw: str) -> None:
    with pytest.raises(WorkspaceError):
        ws.resolve(raw)


@pytest.mark.parametrize(
    "rel",
    [
        ".env",
        ".env.local",
        "app/.env",
        "app/.env.production",
        ".git/config",
        ".git/hooks/pre-commit",
        "certs/server.pem",
        "server.key",
        "id_rsa",
        ".kartrix/logs/kartrix.jsonl",
        ".kartrix/current_session",
    ],
)
def test_secrets_and_internal_state_not_readable(ws: Workspace, rel: str) -> None:
    with pytest.raises(WorkspaceError, match="protected"):
        ws.resolve(rel, "read")


@pytest.mark.parametrize(
    "rel", [".env.example", "app/.env.sample", "id_rsa.pub", ".kartrix/skills/x/SKILL.md", "src/env.py", ".gitignore"]
)
def test_ordinary_files_readable(ws: Workspace, rel: str) -> None:
    assert ws.resolve(rel, "read") == ws.root / rel


@pytest.mark.parametrize("rel", [".git/hooks/pre-commit", ".kartrix/skills/x/SKILL.md", ".env", "deploy.key"])
def test_protected_paths_not_writable(ws: Workspace, rel: str) -> None:
    with pytest.raises(WorkspaceError, match="protected"):
        ws.resolve(rel, "write")


def test_git_directory_itself_protected(ws: Workspace) -> None:
    (ws.root / ".git").mkdir()
    with pytest.raises(WorkspaceError, match="protected"):
        ws.resolve(".git", "read")


@pytest.mark.skipif(not CASE_INSENSITIVE, reason="case-insensitive file systems only")
def test_protection_ignores_case(ws: Workspace) -> None:
    for rel in (".ENV", ".Git/config", "Server.PEM"):
        with pytest.raises(WorkspaceError, match="protected"):
            ws.resolve(rel, "read")


def test_link_to_outside_blocked(ws: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    _link(outside, ws.root / "escape")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve("escape/secret.txt")
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        ws.resolve("escape/new.txt", "write")


def test_link_cannot_disguise_protected_path(ws: Workspace) -> None:
    (ws.root / ".git").mkdir()
    _link(ws.root / ".git", ws.root / "innocent")
    with pytest.raises(WorkspaceError, match="protected"):
        ws.resolve("innocent/config")


def test_link_inside_workspace_allowed(ws: Workspace) -> None:
    (ws.root / "real").mkdir()
    _link(ws.root / "real", ws.root / "alias")
    assert ws.resolve("alias/a.txt") == ws.root / "real" / "a.txt"


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        Workspace.create(tmp_path / "missing")


@pytest.mark.parametrize(
    "raw",
    [
        r"\\server\share\x.txt",
        "//server/share/x.txt",
        r"\\?\C:\Windows\x",
        r"\\.\PhysicalDrive0",
        "C:foo.txt",
        "notes.txt:hidden",
        "a.txt::$DATA",
        "nul",
        "src/COM1.txt",
        "lpt9",
        ".env.",
        "dir /file",
        "file.txt ",
    ],
)
def test_windows_path_tricks_rejected(raw: str) -> None:
    with pytest.raises(WorkspaceError):
        check_windows_path(raw)


@pytest.mark.parametrize("raw", [r"C:\work\app\main.py", "src/app.py", r"src\app.py", "..", "./a", "console.py"])
def test_windows_normal_paths_pass(raw: str) -> None:
    check_windows_path(raw)


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_windows_checks_applied_on_windows(ws: Workspace) -> None:
    for raw in (".env.", "a.txt:stream", "nul"):
        with pytest.raises(WorkspaceError):
            ws.resolve(raw)
