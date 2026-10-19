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
    assert lifecycle.update() == 0
    assert handed_off == [["uv", "tool", "upgrade", "kartrix"]]


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
