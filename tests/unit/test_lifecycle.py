"""``kartrix update`` / ``kartrix uninstall`` (kartrix.lifecycle) — uv and process hand-off are faked; nothing is
really upgraded or removed, and only a temporary KARTRIX_HOME is ever deleted."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kartrix import lifecycle


def test_installed_as_tool(tmp_path: Path) -> None:
    tools = tmp_path / "uv-tools"
    (tools / "kartrix").mkdir(parents=True)
    assert lifecycle.installed_as_tool(tools, prefix=tools / "kartrix")
    assert not lifecycle.installed_as_tool(tools, prefix=tmp_path / "checkout" / ".venv")  # a development checkout
    assert not lifecycle.installed_as_tool(tools, prefix=tools / "other-tool")
    assert not lifecycle.installed_as_tool(None, prefix=tools / "kartrix")  # uv couldn't say


def test_handoff_command() -> None:
    windows = lifecycle.handoff_command("C:/uv/uv.exe", ["tool", "upgrade", "kartrix"], "win32")
    assert windows[:2] == ["cmd.exe", "/c"]
    assert "timeout /t 3" in windows[2] and '"C:/uv/uv.exe" tool upgrade kartrix' in windows[2]  # after Kartrix exits
    spaced = lifecycle.handoff_command("uv", ["--constraints", "C:/My Repo/constraints.txt"], "win32")
    assert '"C:/My Repo/constraints.txt"' in spaced[2]  # a path with spaces stays one argument
    assert lifecycle.handoff_command("/usr/bin/uv", ["tool", "uninstall", "kartrix"], "linux") == [
        "/usr/bin/uv", "tool", "uninstall", "kartrix",
    ]  # fmt: skip


@pytest.fixture
def handed_off(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(lifecycle, "_hand_off", lambda uv, args: calls.append([uv, *args]) or 0)
    return calls


def test_update_without_uv(monkeypatch: pytest.MonkeyPatch, handed_off: list[Any], capsys: Any) -> None:
    monkeypatch.setattr(lifecycle, "uv_executable", lambda: None)
    assert lifecycle.update() == 1
    assert "uv isn't on PATH" in capsys.readouterr().out and handed_off == []


def test_update_from_a_checkout(monkeypatch: pytest.MonkeyPatch, handed_off: list[Any], capsys: Any) -> None:
    monkeypatch.setattr(lifecycle, "uv_executable", lambda: "uv")
    monkeypatch.setattr(lifecycle, "tool_dir", lambda uv: Path("/nowhere/uv-tools"))
    assert lifecycle.update() == 1
    assert "git pull" in capsys.readouterr().out and handed_off == []


def test_update_a_tool_install(monkeypatch: pytest.MonkeyPatch, handed_off: list[Any]) -> None:
    monkeypatch.setattr(lifecycle, "uv_executable", lambda: "uv")
    monkeypatch.setattr(lifecycle, "tool_dir", lambda uv: Path("/x"))
    monkeypatch.setattr(lifecycle, "installed_as_tool", lambda tools: True)
    monkeypatch.delenv("KARTRIX_SOURCE", raising=False)
    monkeypatch.delenv("KARTRIX_CONSTRAINTS", raising=False)
    assert lifecycle.update() == 0
    # a reinstall with the published constraints, not `uv tool upgrade` (which keeps the first install's pins)
    assert handed_off == [
        ["uv", "tool", "install", "--python", "3.12", "--force", "--refresh-package", "kartrix",
         "--constraints", lifecycle.PUBLISHED_CONSTRAINTS, lifecycle.PUBLISHED_SOURCE],
    ]  # fmt: skip


def test_constraints_for_each_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KARTRIX_CONSTRAINTS", raising=False)
    assert lifecycle.constraints_for(lifecycle.PUBLISHED_SOURCE) == lifecycle.PUBLISHED_CONSTRAINTS
    (tmp_path / "constraints.txt").write_text("rich==14.0.0\n", encoding="utf-8")
    assert lifecycle.constraints_for(str(tmp_path)) == str(tmp_path / "constraints.txt")  # a checkout's own
    assert lifecycle.constraints_for(str(tmp_path / "kartrix-0.1.0-py3-none-any.whl")) is None
    assert lifecycle.constraints_for("git+https://example.com/fork") is None
    monkeypatch.setenv("KARTRIX_CONSTRAINTS", "https://example.com/c.txt")
    assert lifecycle.constraints_for("git+https://example.com/fork") == "https://example.com/c.txt"
    monkeypatch.setenv("KARTRIX_CONSTRAINTS", "none")
    assert lifecycle.constraints_for(lifecycle.PUBLISHED_SOURCE) is None
    assert "--constraints" not in lifecycle.install_args(lifecycle.PUBLISHED_SOURCE, None)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(lifecycle, "uv_executable", lambda: "uv")
    monkeypatch.setattr(lifecycle, "tool_dir", lambda uv: Path("/x"))
    monkeypatch.setattr(lifecycle, "installed_as_tool", lambda tools: True)
    monkeypatch.setattr(lifecycle, "stop_built_in_database", lambda: None)
    from kartrix.paths import user_config_dir, user_data_dir

    (user_data_dir() / "postgres").mkdir(parents=True)
    (user_config_dir() / "keys.env").write_text("NVIDIA_API_KEY=x\n", encoding="utf-8")
    return tmp_path / "home"


def test_uninstall_keeps_data_unless_asked(home: Path, handed_off: list[Any]) -> None:
    assert lifecycle.uninstall(ask=lambda prompt: "") == 0
    assert (home / "config" / "keys.env").exists() and (home / "data" / "postgres").exists()
    assert handed_off == [["uv", "tool", "uninstall", "kartrix"]]


def test_uninstall_deletes_data_when_confirmed(
    home: Path, monkeypatch: pytest.MonkeyPatch, handed_off: list[Any]
) -> None:
    stopped: list[bool] = []
    monkeypatch.setattr(lifecycle, "stop_built_in_database", lambda: stopped.append(True))
    assert lifecycle.uninstall(ask=lambda prompt: "y") == 0
    assert stopped == [True]  # the database stops before its folder goes
    assert not (home / "config").exists() and not (home / "data").exists()
    assert handed_off == [["uv", "tool", "uninstall", "kartrix"]]


def test_uninstall_from_a_checkout_never_hands_off(
    home: Path, monkeypatch: pytest.MonkeyPatch, handed_off: list[Any]
) -> None:
    monkeypatch.setattr(lifecycle, "installed_as_tool", lambda tools: False)
    assert lifecycle.uninstall(ask=lambda prompt: "n") == 1
    assert handed_off == []
