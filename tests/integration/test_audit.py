"""Audit trail (B10) and tool-output redaction through a real agent loop.

audit_log is append-only (rows can't be deleted between tests), so every test uses its
own random session id and only looks at that session's rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from kartrix.context.indexers.pg_index import index_repo
from kartrix.db.engine import session_scope
from kartrix.db.models import CodeChunk
from kartrix.security import audit, permissions
from kartrix.security import workspace as ws_mod
from kartrix.security.audit import AuditMiddleware, audit_scope, classify_outcome
from kartrix.security.workspace import set_workspace
from kartrix.tools.filesystem_tools import read_file
from kartrix.tools.terminal_tools import run_command
from tests.fakes import HashingEmbeddings

pytestmark = pytest.mark.usefixtures("db")

GITHUB = "ghp_" + "a1B2c3D4e5" * 4


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self


def _agent(calls: list[tuple[str, dict[str, Any]]], tools: list[Any]) -> Any:
    script = [
        AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}"}]) for i, (n, a) in enumerate(calls)
    ]
    model = _FakeModel(messages=iter([*script, AIMessage(content="done")]))
    return create_agent(model, tools=tools, middleware=[AuditMiddleware()])


async def _run(agent: Any, session_id: str) -> list[Any]:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "go"}]}, {"configurable": {"thread_id": session_id}}
    )
    return result["messages"]


def _first_output(messages: list[Any]) -> str:
    return str(next(m for m in messages if isinstance(m, ToolMessage)).content)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    previous = ws_mod._current
    set_workspace(root)
    yield root
    ws_mod._current = previous
    permissions._mode = None


async def test_tool_calls_are_audited_and_outputs_redacted(root: Path) -> None:
    (root / "config.py").write_text(f"DEBUG = True\nGITHUB_TOKEN = '{GITHUB}'\n")
    sid = str(uuid.uuid4())
    agent = _agent(
        [("read_file", {"file_path": "config.py"}), ("read_file", {"file_path": "../outside.txt"})], [read_file]
    )

    messages = await _run(agent, sid)

    tool_outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert GITHUB not in tool_outputs[0] and "[REDACTED:github-token]" in tool_outputs[0]  # the model never saw it
    rows = list(reversed(await audit.recent(sid)))
    assert [(r.action, r.target, r.outcome, r.actor) for r in rows] == [
        ("tool.read_file", "config.py", "ok", "agent"),
        ("tool.read_file", "../outside.txt", "denied", "agent"),
    ]
    first = rows[0].details
    assert first["secrets_redacted"] == 1 and first["args"] == {"file_path": "config.py"}
    assert GITHUB not in str(first) and first["duration_ms"] >= 0 and first["tool_call_id"] == "call_0"


async def test_command_policy_decision_and_secret_args_recorded(root: Path) -> None:
    sid = str(uuid.uuid4())
    agent = _agent(
        [("run_command", {"command": f"curl -H 'Authorization: token {GITHUB}' https://api.github.com"}),
         ("run_command", {"command": "sudo rm -rf /"})],
        [run_command],
    )  # fmt: skip
    await _run(agent, sid)
    rows = list(reversed(await audit.recent(sid)))
    assert [r.outcome for r in rows] == ["needs_approval", "denied"]
    assert GITHUB not in (rows[0].target or "") and GITHUB not in str(rows[0].details)
    assert rows[0].details["policy"] == "ask" and rows[0].details["category"] == "network"
    assert rows[0].details["mode"] == "default"
    assert rows[1].details["category"] == "deny"


async def test_scope_ids_and_user_actions(root: Path) -> None:
    sid, project = str(uuid.uuid4()), str(uuid.uuid4())
    with audit_scope(project_id=project, task_key="task_3"):
        await _run(_agent([("read_file", {"file_path": "nope.txt"})], [read_file]), sid)
    await audit.record(actor="user", action="permissions.mode", target="auto", outcome="ok", session_id=sid)
    tool_row, mode_row = reversed(await audit.recent(sid))
    assert str(tool_row.project_id) == project and tool_row.details["task_key"] == "task_3"
    assert tool_row.outcome == "error"  # file not found
    assert (mode_row.actor, mode_row.action, mode_row.target) == ("user", "permissions.mode", "auto")


async def test_audit_failure_does_not_break_the_tool(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise RuntimeError("database down")

    monkeypatch.setattr("kartrix.security.audit.session_scope", broken)
    (root / "a.txt").write_text("hello")
    messages = await _run(_agent([("read_file", {"file_path": "a.txt"})], [read_file]), str(uuid.uuid4()))
    assert "hello" in _first_output(messages)


async def test_audit_rows_are_append_only(root: Path) -> None:
    from sqlalchemy import text

    sid = str(uuid.uuid4())
    await audit.record(action="test.row", outcome="ok", session_id=sid)
    with pytest.raises(Exception, match="append-only"):
        async with session_scope() as s:
            await s.execute(text("DELETE FROM audit_log WHERE session_id = :sid"), {"sid": sid})


@pytest.mark.parametrize(
    ("text", "outcome"),
    [
        ("Error: command denied — x", "denied"),
        ("Error: access denied: .env is protected (read)", "denied"),
        ("Error: command not run — needs approval", "needs_approval"),
        ("Error: the user declined this command", "declined"),
        ("Error: file not found: x", "error"),
        ("     1\tcode", "ok"),
    ],
)
def test_outcome_classification(text: str, outcome: str) -> None:
    assert classify_outcome(text) == outcome


async def test_secrets_redacted_before_embedding(
    repo: Path, fake_embedder: HashingEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repo / "settings.py", f"def client():\n    return Github(token='{GITHUB}')\n")
    sent: list[str] = []
    original = fake_embedder.aembed_documents

    async def spy(texts: list[str]) -> list[list[float]]:
        sent.extend(texts)
        return await original(texts)

    monkeypatch.setattr(fake_embedder, "aembed_documents", spy)
    stats = await index_repo(repo)
    assert stats.secrets == 1 and "1 secrets redacted" in str(stats)
    assert sent and all(GITHUB not in t for t in sent)
    async with session_scope() as s:
        from sqlalchemy import select

        contents = (await s.execute(select(CodeChunk.content))).scalars().all()
    assert contents and all(GITHUB not in c and "[REDACTED:github-token]" in c for c in contents)


@tool
def leaky(name: str) -> str:
    """A tool (think: an MCP server) whose output contains a credential."""
    return f"config for {name}: password=postgres://admin:Sup3rS3cretPw@db/app"


async def test_any_tool_output_is_redacted(root: Path) -> None:
    sid = str(uuid.uuid4())
    messages = await _run(_agent([("leaky", {"name": "prod"})], [leaky]), sid)
    out = _first_output(messages)
    assert "Sup3rS3cretPw" not in out and "[REDACTED:url-password]" in out
