"""The first-run key prompt, ``kartrix keys`` and ``kartrix doctor`` — no real provider calls (validate is faked)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from kartrix import cli, doctor, keys


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    for info in keys.KNOWN:
        monkeypatch.delenv(info.name, raising=False)
    return tmp_path / "home"


def test_first_run_without_a_terminal_explains_and_stops(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert not keys.first_run(interactive=False)
    out = capsys.readouterr().out
    assert "kartrix keys set NVIDIA_API_KEY" in out and "build.nvidia.com" in out


def test_first_run_asks_checks_and_saves(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["bad-key", "nvapi-good-key-123456"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    monkeypatch.setattr(keys, "validate", lambda name, value: (value.startswith("nvapi-"), "rejected"))
    assert keys.first_run(interactive=True)
    assert keys.read() == {"NVIDIA_API_KEY": "nvapi-good-key-123456"}
    assert keys.missing_required() == []
    assert keys.first_run(interactive=True)  # nothing missing: no questions


def test_first_run_cancel(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
    assert not keys.first_run(interactive=True)
    assert keys.read() == {}


def test_keys_command(home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(keys, "validate", lambda name, value: (True, ""))
    monkeypatch.setattr("sys.stdin", io.StringIO("gsk_from_stdin_123456\n"))
    assert cli.main(["keys", "set", "GROQ_API_KEY"]) == 0
    assert keys.read()["GROQ_API_KEY"] == "gsk_from_stdin_123456"
    assert cli.main(["keys"]) == 0
    listing = capsys.readouterr().out
    assert "gsk_fr" in listing and "gsk_from_stdin_123456" not in listing  # masked
    assert "NVIDIA_API_KEY" in listing and "MISSING" in listing
    monkeypatch.setattr(keys, "validate", lambda name, value: (False, "the provider rejected this key"))
    monkeypatch.setattr("sys.stdin", io.StringIO("wrong\n"))
    assert cli.main(["keys", "set", "OPENROUTER_API_KEY"]) == 1
    assert "OPENROUTER_API_KEY" not in keys.read()  # a rejected key is not saved
    assert cli.main(["keys", "remove", "GROQ_API_KEY"]) == 0
    assert cli.main(["keys", "remove", "GROQ_API_KEY"]) == 1


def test_doctor_reports_and_exits_by_required_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = doctor.Check("ok", "Thing", "fine — really")
    monkeypatch.setattr(
        doctor,
        "check_keys",
        lambda online=True: [doctor.Check("fail", "NVIDIA key", "not set", "kartrix keys set NVIDIA_API_KEY")],
    )
    for name in ("check_database", "check_redis", "check_sandbox", "check_git", "check_disk"):
        monkeypatch.setattr(doctor, name, lambda: ok)
    console = Console(record=True, width=200, color_system=None)
    assert doctor.run_doctor(console) == 1
    text = console.export_text()
    assert "[FAIL] NVIDIA key: not set" in text and "-> kartrix keys set NVIDIA_API_KEY" in text
    assert "fine - really" in text  # ASCII only
    monkeypatch.setattr(doctor, "check_keys", lambda online=True: [ok])
    assert doctor.run_doctor(Console(record=True, width=200)) == 0


def test_doctor_key_checks(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-abcdefghijklmn")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghijklmnop")
    verdicts: dict[str, Any] = {"NVIDIA_API_KEY": (True, ""), "GROQ_API_KEY": (False, "the provider rejected this key")}
    monkeypatch.setattr(keys, "validate", lambda name, value: verdicts[name])
    by_name = {c.name: c for c in doctor.check_keys(online=True)}
    assert by_name["NVIDIA key"].status == "ok"
    assert by_name["Groq key"].status == "warn" and "GROQ_API_KEY" in by_name["Groq key"].fix
    assert by_name["OpenRouter key"].status == "off"  # optional and unset
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert {c.name: c for c in doctor.check_keys(online=False)}["NVIDIA key"].status == "fail"


def test_doctor_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(0, 0, int(1.2 * 2**30)))
    check = doctor.check_disk()
    assert check.status == "warn" and "KARTRIX_HOME" in check.fix
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(0, 0, 50 * 2**30))
    assert doctor.check_disk().status == "ok"
