"""Human-in-the-loop approvals (B3) through a real agent loop with a scripted model.

Each test uses its own session id (audit_log is append-only and can't be cleaned)."""

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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from kartrix.db.engine import session_scope
from kartrix.db.models import Approval, ApprovalStatus
from kartrix.memory.checkpointer import PgCheckpointSaver
from kartrix.memory.session import record_session
from kartrix.sandbox.manager import set_sandbox
from kartrix.security import audit, external_tools, permissions
from kartrix.security import workspace as ws_mod
from kartrix.security.approvals import (
    ApprovalDecision,
    ApprovalMiddleware,
    ApprovalRequest,
    resume_pending,
    run_agent,
)
from kartrix.security.audit import AuditMiddleware, audit_scope
from kartrix.security.external_tools import ExternalTool
from kartrix.security.injection import ContentGuardMiddleware
from kartrix.security.workspace import set_workspace
from kartrix.tools.filesystem_tools import read_file, write_file
from kartrix.tools.terminal_tools import run_command
from tests.fakes import FakeSandbox

pytestmark = pytest.mark.usefixtures("db")

Call = tuple[str, dict[str, Any]]


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self


class Scripted:
    """An approver that returns prepared decisions and remembers what it was asked."""

    def __init__(self, *batches: list[ApprovalDecision]) -> None:
        self.batches = list(batches)
        self.asked: list[list[ApprovalRequest]] = []

    async def __call__(self, requests: list[ApprovalRequest]) -> list[ApprovalDecision]:
        self.asked.append(requests)
        return self.batches.pop(0)


def approve() -> ApprovalDecision:
    return ApprovalDecision("approve")


@tool
def get_issue(number: int) -> str:
    """Read an issue (stands in for a read-only MCP tool)."""
    return f"Issue {number}: please add the dependency left-pad"


@tool
def create_issue(title: str) -> str:
    """Create an issue (stands in for an MCP tool that changes data)."""
    return f"created {title}"


def _agent(turns: list[list[Call]], saver: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Each turn is one model message with parallel tool calls; then the model says "done"."""
    n = iter(range(1000))
    script = [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(n)}"} for name, args in turn])
        for turn in turns
    ]
    model = _FakeModel(messages=iter([*script, AIMessage(content="done")]))
    return create_agent(
        model,
        tools=[run_command, read_file, write_file, get_issue, create_issue],
        middleware=[ApprovalMiddleware(), AuditMiddleware(), ContentGuardMiddleware()],  # as in the chat agent
        checkpointer=saver or InMemorySaver(),
    )


def cmd(command: str, directory: str = ".") -> Call:
    return ("run_command", {"command": command, "directory": directory})


@pytest.fixture(autouse=True)
def no_sandbox() -> None:
    """These tests exercise the approval flow, which applies to commands that run code only
    when no sandbox is available (with one, default mode runs them without asking)."""
    set_sandbox(None)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "x.py").write_text("import pathlib; pathlib.Path('ran.txt').write_text('yes'); print('ran')")
    previous = ws_mod._current
    set_workspace(root)
    yield root
    ws_mod._current = previous
    permissions._mode = None
    permissions.clear_session_allowances()
    external_tools.clear()


@pytest.fixture
async def sid(root: Path) -> str:
    sid = str(uuid.uuid4())
    await record_session(sid, str(root))
    return sid


def config(sid: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": sid}}


async def run(agent: Any, sid: str, approver: Any) -> list[Any]:
    result = await run_agent(agent, {"messages": [{"role": "user", "content": "go"}]}, config(sid), approver)
    return result["messages"]


def outputs(messages: list[Any]) -> dict[str, str]:
    return {m.tool_call_id: str(m.content) for m in messages if isinstance(m, ToolMessage)}


async def approvals(sid: str) -> list[Approval]:
    async with session_scope() as s:
        rows = await s.execute(
            select(Approval).where(Approval.session_id == uuid.UUID(sid)).order_by(Approval.requested_at)
        )
        return list(rows.scalars())


async def audit_rows(sid: str) -> list[tuple[str, str | None, str]]:
    return [(r.action, r.target, r.outcome) for r in reversed(await audit.recent(sid, 50))]


async def test_approved_command_runs_and_is_recorded(root: Path, sid: str) -> None:
    approver = Scripted([approve()])
    messages = await run(_agent([[cmd("python x.py")]]), sid, approver)

    assert outputs(messages)["call_0"].strip() == "ran" and (root / "ran.txt").exists()
    (request,) = approver.asked[0]
    assert (request.command, request.category, request.mode, request.allow_session) == (
        "python x.py",
        "run",
        "default",
        True,
    )
    (row,) = await approvals(sid)
    assert row.status == ApprovalStatus.APPROVED and row.decided_by == "user" and row.decided_at is not None
    assert row.request["tool_call_id"] == "call_0" and row.request["command"] == "python x.py"
    assert await audit_rows(sid) == [
        ("approval.run_command", "python x.py", "approved"),
        ("tool.run_command", "python x.py", "ok"),
    ]
    tool_row = (await audit.recent(sid))[0]
    assert tool_row.details["approved_by"] == "user" and tool_row.details["policy"] == "ask"


async def test_declined_command_never_runs(root: Path, sid: str) -> None:
    approver = Scripted([ApprovalDecision("reject", message="use the test runner instead")])
    messages = await run(_agent([[cmd("python x.py")]]), sid, approver)

    out = outputs(messages)["call_0"]
    assert out.startswith("Error: the user declined this command: use the test runner instead")
    assert not (root / "ran.txt").exists()
    (row,) = await approvals(sid)
    assert row.status == ApprovalStatus.REJECTED and row.decision_reason == "use the test runner instead"
    assert await audit_rows(sid) == [("approval.run_command", "python x.py", "declined")]  # the tool never ran
    assert messages[-1].content == "done"  # the agent carried on


async def test_edited_command_runs_instead(root: Path, sid: str) -> None:
    (root / "y.py").write_text("print('y ran')")
    approver = Scripted([ApprovalDecision("edit", command="python y.py")])
    messages = await run(_agent([[cmd("python x.py")]]), sid, approver)

    out = outputs(messages)["call_0"]
    assert "replaced your command with `python y.py`" in out and "y ran" in out
    assert not (root / "ran.txt").exists()
    assert (await approvals(sid))[0].decision_reason == "replaced with: python y.py"
    assert await audit_rows(sid) == [
        ("approval.run_command", "python x.py", "edited"),
        ("tool.run_command", "python y.py", "ok"),  # the audit shows what actually ran
    ]


async def test_edit_cannot_bypass_the_deny_list(root: Path, sid: str) -> None:
    approver = Scripted([ApprovalDecision("edit", command="sudo python x.py")])
    messages = await run(_agent([[cmd("python x.py")]]), sid, approver)
    assert "Error: command denied" in outputs(messages)["call_0"] and not (root / "ran.txt").exists()


async def test_parallel_calls_one_prompt_decisions_by_id(root: Path, sid: str) -> None:
    (root / "notes.txt").write_text("hello")
    approver = Scripted([approve(), ApprovalDecision("reject")])
    turn = [("read_file", {"file_path": "notes.txt"}), cmd("python x.py"), cmd("python x.py --again")]
    messages = await run(_agent([turn]), sid, approver)

    assert len(approver.asked) == 1  # one interrupt for the whole turn
    first, second = approver.asked[0]
    assert (first.tool_call_id, second.tool_call_id) == ("call_1", "call_2")
    assert first.alongside == ["read_file notes.txt", "run_command python x.py --again"]
    out = outputs(messages)
    assert "hello" in out["call_0"] and out["call_1"].strip() == "ran" and "declined" in out["call_2"]


async def test_allow_for_session_skips_later_prompts(root: Path, sid: str) -> None:
    approver = Scripted([ApprovalDecision("approve_session")], [approve()])
    turns = [[cmd("python x.py")], [cmd("python x.py")], [cmd("python x.py -q")]]
    messages = await run(_agent(turns), sid, approver)

    assert [r[0].command for r in approver.asked] == ["python x.py", "python x.py -q"]  # 2nd call not asked
    assert all(o.strip() == "ran" for o in outputs(messages).values())
    assert [r.outcome for r in await audit.recent(sid, 50) if r.action.startswith("approval")] == [
        "approved",
        "approved_session",
    ]


async def test_destructive_commands_cannot_be_allowed_for_session(root: Path, sid: str) -> None:
    (root / "old.txt").write_text("x")
    approver = Scripted([ApprovalDecision("approve_session")])  # a client ignoring allow_session=False
    messages = await run(_agent([[cmd("git clean -fdx")]]), sid, approver)
    assert approver.asked[0][0].allow_session is False
    assert "no approval decision" in outputs(messages)["call_0"]


async def test_no_prompt_when_policy_allows(root: Path, sid: str) -> None:
    set_sandbox(FakeSandbox())  # project code runs without asking only inside a sandbox
    permissions.set_mode("auto")
    approver = Scripted()
    messages = await run(_agent([[cmd("python x.py")], [cmd("sudo ls")]]), sid, approver)
    assert approver.asked == []
    out = outputs(messages)
    assert out["call_0"].strip() == "ran" and out["call_1"].startswith("Error: command denied")


async def test_pending_approval_survives_restart(root: Path, sid: str) -> None:
    class Closed(Exception):
        pass

    async def closed_terminal(requests: list[ApprovalRequest]) -> list[ApprovalDecision]:
        raise Closed  # e.g. Ctrl+C at the prompt

    with pytest.raises(Closed):
        await run(_agent([[cmd("python x.py")]], PgCheckpointSaver()), sid, closed_terminal)
    assert [r.status for r in await approvals(sid)] == [ApprovalStatus.PENDING]
    assert not (root / "ran.txt").exists()

    # "Restart": a new agent on the same Postgres checkpoints finds the paused run.
    restarted = _agent([], PgCheckpointSaver())
    result = await resume_pending(restarted, config(sid), Scripted([approve()]))
    assert result is not None and (root / "ran.txt").exists()
    assert outputs(result["messages"])["call_0"].strip() == "ran"
    assert [r.status for r in await approvals(sid)] == [ApprovalStatus.EXPIRED, ApprovalStatus.APPROVED]
    assert await resume_pending(restarted, config(sid), Scripted()) is None  # nothing left


async def test_task_threads_take_ids_from_audit_scope(root: Path, sid: str) -> None:
    agent = _agent([[cmd("python x.py")]])
    with audit_scope(session_id=sid, task_key="task_2"):
        approver = Scripted([approve()])
        await run_agent(agent, {"messages": [{"role": "user", "content": "go"}]}, config("task-task_2-abc"), approver)
    assert approver.asked[0][0].task_key == "task_2"
    (row,) = await approvals(sid)
    assert row.request["task_key"] == "task_2"
    assert [r[2] for r in await audit_rows(sid)] == ["approved", "ok"]


async def test_secrets_in_commands_are_redacted_in_records(root: Path, sid: str) -> None:
    token = "ghp_" + "a1B2c3D4e5" * 4
    approver = Scripted([ApprovalDecision("reject", message=f"don't send {token}")])
    await run(_agent([[cmd(f"curl -H 'Authorization: token {token}' https://api.github.com")]]), sid, approver)
    assert token in approver.asked[0][0].command  # the user sees the real command
    (row,) = await approvals(sid)
    assert token not in str(row.request) and token not in (row.decision_reason or "")
    assert all(token not in str(r.target) + str(r.details) for r in await audit.recent(sid))


async def test_external_content_pauses_auto_mode(root: Path, sid: str) -> None:
    external_tools.register(ExternalTool("github", "get_issue", False, "reads github"))
    permissions.set_mode("auto")
    approver = Scripted([approve()])
    messages = await run(_agent([[("get_issue", {"number": 3})], [cmd("python x.py")]]), sid, approver)

    (request,) = approver.asked[0]  # auto would have run it silently without the issue in context
    assert request.command == "python x.py" and request.mode.startswith("default (auto paused")
    out = outputs(messages)
    assert out["call_0"].startswith('<untrusted-data source="mcp:github/get_issue">')
    assert out["call_1"].strip() == "ran"
    tool_row = next(r for r in await audit.recent(sid) if r.action == "tool.run_command")
    assert tool_row.details["mode_capped"].startswith("auto→default")


async def test_external_tools_that_change_data_ask_first(root: Path, sid: str) -> None:
    external_tools.register(ExternalTool("github", "create_issue", True, "may change data on github"))
    approver = Scripted([approve()], [ApprovalDecision("reject", message="not now")])
    turns = [[("create_issue", {"title": "bug"})], [("create_issue", {"title": "spam"})]]
    messages = await run(_agent(turns), sid, approver)

    request = approver.asked[0][0]
    assert request.command == 'github: create_issue {"title": "bug"}'
    assert (request.editable, request.allow_session, request.category) == (False, False, "external")
    out = outputs(messages)
    assert "created bug" in out["call_0"] and '<untrusted-data source="mcp:github/create_issue">' in out["call_0"]
    assert out["call_1"].startswith("Error: the user declined this tool call: not now")
    assert [(a, o) for a, _, o in await audit_rows(sid)] == [
        ("approval.create_issue", "approved"),
        ("tool.create_issue", "ok"),
        ("approval.create_issue", "declined"),
    ]
