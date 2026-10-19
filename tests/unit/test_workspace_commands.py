"""Workspace commands (kartrix.core.commands) — parsing, running them against a fake core, the shared terminal
flow (kartrix.ui.command_views) — and their way through a server: the worker pool and the worker process."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from kartrix.core import commands
from kartrix.security import checkpoints, permissions
from kartrix.security.checkpoints import Entry, UndoResult
from kartrix.server.protocol import CommandOp
from kartrix.server.workers import PoolError, WorkerPool, WorkerProcess
from kartrix.ui import command_views


class Host:
    session_id = "s"
    mcp = None

    def __init__(self) -> None:
        self.rebuilt = 0

    def rebuild_agent(self) -> None:
        self.rebuilt += 1


@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from kartrix.security import audit

    rows: list[dict[str, Any]] = []

    async def record(**row: Any) -> None:
        rows.append(row)

    monkeypatch.setattr(audit, "record", record)
    return rows


@pytest.fixture(autouse=True)
def default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permissions, "_mode", None)


def _console() -> Console:
    return Console(record=True, width=200, color_system=None)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/undo", ("undo", {"force": False})),
        ("/redo force", ("redo", {"force": True})),
        ("/checkpoints", ("checkpoints", {})),
        ("/mode", ("mode", {"target": None})),
        ("/mode auto", ("mode", {"target": "auto"})),
        ("/memory", ("memory.list", {})),
        ("/memory add --user be brief", ("memory.add", {"content": "be brief", "user": True})),
        ("/memory add tests use pytest", ("memory.add", {"content": "tests use pytest", "user": False})),
        ("/memory forget 1a2b", ("memory.forget", {"id": "1a2b"})),
        ("/skills", ("skills.list", {})),
        ("/skills trust lint", ("skills.trust", {"name": "lint"})),
        ("/skills untrust lint", ("skills.untrust", {"name": "lint"})),
        ("/mcp", ("mcp.list", {})),
        ("/connect", ("mcp.list", {})),
        ("/connect github", ("mcp.connect", {"server": "github"})),
        ("/disconnect github", ("mcp.disconnect", {"server": "github"})),
    ],
)
def test_parse(line: str, expected: tuple[str, dict[str, Any]]) -> None:
    assert commands.is_command(line)
    assert commands.parse(line) == expected


@pytest.mark.parametrize("line", ["/undo now", "/memory add", "/memory add --user", "/memory drop x", "/skills nuke x"])
def test_parse_rejects_bad_usage(line: str) -> None:
    with pytest.raises(commands.CommandError, match="Usage"):
        commands.parse(line)


def test_not_workspace_commands() -> None:
    assert not any(commands.is_command(line) for line in ("", "/ask x", "/model", "/undone", "undo the change"))


def test_every_command_has_a_handler_and_a_view() -> None:
    assert set(commands.COMMANDS) == set(commands._HANDLERS) == set(command_views._RENDERERS)


async def test_mode_lists_changes_and_audits(audited: list[dict[str, Any]]) -> None:
    host = Host()
    assert (await commands.run(host, "mode", {"target": None}))["changed"] is False
    result = await commands.run(host, "mode", {"target": "read_only"})
    assert (result["mode"], result["changed"]) == ("read_only", True)
    assert permissions.get_mode() == "read_only"
    assert audited[-1]["action"] == "permissions.mode" and audited[-1]["details"]["previous"] == "default"
    with pytest.raises(commands.CommandError, match="unknown permission mode"):
        await commands.run(host, "mode", {"target": "yolo"})


async def test_mcp_commands_say_when_mcp_is_off() -> None:
    with pytest.raises(commands.CommandError, match="MCP servers are off"):
        await commands.run(Host(), "mcp.list", {})


async def test_unknown_command_name() -> None:
    with pytest.raises(commands.CommandError, match="Unknown command"):
        await commands.run(Host(), "rm -rf", {})


class FakeStore:
    """A checkpoint store whose undo is blocked by a later change unless forced."""

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.entry = Entry("e1", "turn: add a --json flag", "a", "b", 1_700_000_000.0, file_count=2)

    def entries(self) -> tuple[list[Entry], list[Entry]]:
        return [self.entry], []

    def undo(self, force: bool = False) -> UndoResult:
        self.calls.append(force)
        if not force:
            return UndoResult(self.entry, applied=False, conflicts=["src/cli.py"])
        return UndoResult(self.entry, applied=True, restored=["src/cli.py"], conflicts=["src/cli.py"])

    def redo(self, force: bool = False) -> None:
        return None


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(checkpoints, "current", lambda: fake)
    return fake


async def test_undo_asks_before_overwriting_later_changes(store: FakeStore, audited: list[dict[str, Any]]) -> None:
    host, console = Host(), _console()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await commands.run(host, name, args)
        json.dumps(result)  # what a server sends must be JSON
        return result

    assert await command_views.execute(console, "/undo", call, interactive=False) is None
    assert store.calls == [False] and audited == []
    assert "Run /undo force" in console.export_text()

    console = _console()
    command_views.Confirm.ask = staticmethod(lambda *a, **k: True)  # type: ignore[method-assign]
    try:
        assert await command_views.execute(console, "/undo", call, interactive=True) == "undo"
    finally:
        del command_views.Confirm.ask
    assert store.calls == [False, False, True]
    text = console.export_text()
    assert "Undid: turn: add a --json flag" in text and "kept in the checkpoint store" in text
    assert audited[-1]["action"] == "checkpoint.undo"


async def test_redo_with_nothing_to_redo_and_checkpoints_listing(store: FakeStore) -> None:
    console = _console()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return await commands.run(Host(), name, args)

    assert await command_views.execute(console, "/redo", call) is None
    assert await command_views.execute(console, "/checkpoints", call) is None
    text = console.export_text()
    assert "Nothing to redo." in text and "turn: add a --json flag" in text


async def test_checkpoints_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "current", lambda: None)
    monkeypatch.setattr(checkpoints, "unavailable_reason", lambda: "not a git repository")
    console = _console()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return await commands.run(Host(), name, args)

    await command_views.execute(console, "/undo", call)
    assert "Checkpoints are off: not a git repository" in console.export_text()


async def test_errors_from_the_server_are_shown() -> None:
    console = _console()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise commands.CommandError("a run is in progress in this workspace — try again when it ends")

    assert await command_views.execute(console, "/memory add x", call) is None
    assert "a run is in progress" in console.export_text()


async def test_skills_trust_rebuilds_the_agent(monkeypatch: pytest.MonkeyPatch, audited: list[dict[str, Any]]) -> None:
    from kartrix.skills import skill_tools
    from kartrix.skills.registry import SkillNotFoundError

    class Registry:
        def trust(self, name: str) -> Any:
            if name != "lint":
                raise SkillNotFoundError(f"No skill named {name!r}")
            return type("Skill", (), {"digest": "ab" * 32})()

        def untrust(self, name: str) -> None:
            pass

    monkeypatch.setattr(skill_tools, "get_registry", Registry)
    host = Host()
    result = await commands.run(host, "skills.trust", {"name": "lint"})
    assert result == {"name": "lint", "trusted": True, "digest": "ab" * 32} and host.rebuilt == 1
    with pytest.raises(commands.CommandError, match="No skill"):
        await commands.run(host, "skills.trust", {"name": "nope"})
    assert host.rebuilt == 1 and [a["action"] for a in audited] == ["skills.trust"]


# ── through a server: the pool ─────────────────────────────────────────


class ScriptedWorker(WorkerProcess):
    """A worker process that answers commands at once, as the real one would over stdout."""

    def __init__(self, pool: WorkerPool, workspace_id: uuid.UUID, answer: dict[str, Any]) -> None:
        super().__init__(pool, workspace_id, uuid.uuid4(), Path("."))
        self.ready.set_result(None)
        self.answer = answer
        self.sent: list[Any] = []

    def send(self, op: Any) -> None:
        self.sent.append(op)
        if isinstance(op, CommandOp):
            reply = {"kind": "command_result", "request_id": op.request_id, **self.answer}
            asyncio.get_running_loop().call_soon(self._on_line, json.dumps(reply).encode())


async def _command(pool: WorkerPool, workspace_id: uuid.UUID, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return await pool.command(
        user_id=uuid.uuid4(), workspace_id=workspace_id, root=Path("."), session_id="s", name=name, args=args
    )


async def test_pool_routes_commands_and_remembers_the_mode(tmp_path: Path) -> None:
    pool, ws = WorkerPool(tmp_path), uuid.uuid4()
    worker = ScriptedWorker(pool, ws, {"ok": True, "result": {"mode": "auto", "modes": [], "changed": True}})
    pool.workers[ws] = worker
    assert (await _command(pool, ws, "mode", {"target": "auto"}))["mode"] == "auto"
    assert worker.sent[0].name == "mode" and worker.sent[0].args == {"target": "auto"}
    assert pool.modes[ws] == "auto" and worker.commands == {}  # a restarted worker gets --mode auto


async def test_pool_command_errors(tmp_path: Path) -> None:
    pool, ws = WorkerPool(tmp_path), uuid.uuid4()
    pool.workers[ws] = ScriptedWorker(pool, ws, {"ok": False, "error": "No memory with that id."})
    with pytest.raises(PoolError) as failed:
        await _command(pool, ws, "memory.forget", {"id": "x"})
    assert (failed.value.status, failed.value.detail) == (422, "No memory with that id.")
    with pytest.raises(PoolError) as unknown:
        await _command(pool, ws, "shell", {})
    assert unknown.value.status == 422

    pool.workers[ws] = ScriptedWorker(pool, ws, {"ok": False, "busy": True, "error": "a run is in progress"})
    with pytest.raises(PoolError) as busy:
        await _command(pool, ws, "undo", {})
    assert busy.value.status == 409


async def test_pool_refuses_changes_during_a_run_but_not_listings(tmp_path: Path) -> None:
    from kartrix.server.workers import RunFeed

    pool, ws = WorkerPool(tmp_path), uuid.uuid4()
    worker = ScriptedWorker(pool, ws, {"ok": True, "result": {"servers": []}})
    worker.feed = RunFeed(uuid.uuid4(), ws)  # a run in progress
    pool.workers[ws] = worker
    with pytest.raises(PoolError) as refused:
        await _command(pool, ws, "undo", {"force": False})
    assert refused.value.status == 409 and worker.sent == []
    assert await _command(pool, ws, "mcp.list", {}) == {"servers": []}


async def test_a_worker_exit_fails_open_commands(tmp_path: Path) -> None:
    pool, ws = WorkerPool(tmp_path), uuid.uuid4()
    worker = ScriptedWorker(pool, ws, {})
    answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    worker.commands["r1"] = answer
    worker._on_exit(1)
    with pytest.raises(PoolError, match="exited"):
        await answer


# ── through a server: the worker process ──────────────────────────────


async def test_worker_answers_commands(monkeypatch: pytest.MonkeyPatch, audited: list[dict[str, Any]]) -> None:
    from kartrix.core.session import CoreSession
    from kartrix.server import worker as worker_mod

    class FakeCore(Host):
        last_budget = None

        async def close(self) -> None:
            pass

    async def start(*args: Any, **kwargs: Any) -> FakeCore:
        return FakeCore()

    monkeypatch.setattr(CoreSession, "start", staticmethod(start))
    ops = [
        CommandOp(request_id="r1", name="mode", args={"target": "read_only"}),
        CommandOp(request_id="r2", name="mcp.list", args={}),
    ]
    lines = b"".join(json.dumps(op.model_dump()).encode() + b"\n" for op in ops)
    out = io.StringIO()

    source = io.BytesIO(lines)  # input ends after these ops: the worker answers them, then shuts down
    assert await worker_mod._serve(worker_mod.Channel(out), source, "s", mcp=False, mode="auto") == 0
    replies = {
        m["request_id"]: m for m in map(json.loads, out.getvalue().splitlines()) if m["kind"] == "command_result"
    }
    assert replies["r1"]["ok"] and replies["r1"]["result"]["mode"] == "read_only"
    assert not replies["r2"]["ok"] and "MCP servers are off" in replies["r2"]["error"]
    assert audited[0]["details"]["previous"] == "auto"  # --mode auto applied at start, before the command
