"""Headless runs end to end (C1, K1): spec → CoreSession → policy approvals → events → report.

The model is scripted (no LLM call) and the embedder is the hashing fake; everything else is real:
Postgres checkpointer, command policy, approvals, audit, budgets, checkpoints, task store.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from kartrix.agent.factory import agent_middleware
from kartrix.core.events import collecting
from kartrix.headless import runner
from kartrix.headless.spec import load_spec
from kartrix.sandbox.manager import set_sandbox
from kartrix.security import permissions
from kartrix.security import workspace as ws_mod
from kartrix.security.approvals import ApprovalMiddleware
from kartrix.tasks.planner import ExecutionPlan, PlannedTask
from kartrix.tasks.task_store import TaskType
from kartrix.tools.filesystem_tools import read_file, write_file
from kartrix.tools.terminal_tools import run_command

pytestmark = pytest.mark.usefixtures("db", "fake_embedder")


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self


def _scripted_agent(commands: list[str]) -> Any:
    """First model turn runs ``commands`` in parallel, then the model answers "done"."""
    calls = [{"name": "run_command", "args": {"command": c}, "id": f"call_{i}"} for i, c in enumerate(commands)]
    model = _FakeModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="done: it ran")]))

    def build(checkpointer: Any, extra_tools: Any = ()) -> Any:
        return create_agent(
            model,
            tools=[run_command, read_file, write_file],
            middleware=agent_middleware(ApprovalMiddleware()),
            checkpointer=checkpointer,
        )

    return build


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "x.py").write_text("import pathlib; pathlib.Path('ran.txt').write_text('yes'); print('ran')\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("kartrix.core.session.get_llm", lambda: None)
    monkeypatch.setattr("kartrix.core.session.get_embedder", lambda: None)
    set_sandbox(None)  # no sandbox: running x.py needs an approval, which the spec's policy gives
    previous = ws_mod._current
    yield root
    ws_mod._current = previous
    permissions._mode = None
    permissions.clear_session_allowances()


def _spec(ws: Path, text: str) -> Any:
    (ws.parent / "spec.yaml").write_text(text, encoding="utf-8")
    return load_spec(ws.parent / "spec.yaml")


async def test_ask_run_with_policy_approvals(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kartrix.core.session.build_agent", _scripted_agent(["python x.py", "curl https://example.com"])
    )
    spec = _spec(ws, "task: run the script\napprovals: {allow: ['python x.py']}\n")
    with collecting() as seen:
        outcome = await runner._run(spec)
    assert outcome.status == "completed" and outcome.answer == "done: it ran"
    assert (ws / "ran.txt").read_text() == "yes"  # approved by the policy, so it ran
    assert "ran.txt" in outcome.files_changed and "x.py" not in outcome.files_changed

    report = runner.build_report(spec, "spec.yaml", outcome, seen, 0.0)
    assert report["exit_code"] == 0 and report["answer"] == "done: it ran"
    decisions = {a["command"]: (a["decision"], a["by"]) for a in report["approvals"]}
    assert decisions == {"python x.py": ("approve", "policy"), "curl https://example.com": ("reject", "policy")}
    # A declined command never reaches the tool: it is in "approvals", not in the tool-call counts.
    assert report["tool_calls"] == {"total": 1, "by_tool": {"run_command": 1}, "by_outcome": {"ok": 1}}
    assert report["usage"]["tool_calls"] == 1 and report["files_changed"] == ["ran.txt"]
    assert set(report["startup"]) >= {"llm", "agent", "index"}  # startup phases, for the evals
    started = next(e for e in seen if e.type == "tool_call_started")
    finished = next(e for e in seen if e.type == "tool_call_finished")
    assert started.args == {"command": "python x.py"} and finished.output_chars

    kinds = [e.type for e in seen]
    assert kinds.index("run_started") < kinds.index("approval_requested") < kinds.index("tool_call_started")
    assert kinds[-1] == "run_finished" and "assistant_message" in kinds
    run_ids = {e.run_id for e in seen if e.type != "notice"}
    assert len(run_ids) == 1 and None not in run_ids  # every event of the run carries its id
    json.dumps([e.model_dump(mode="json") for e in seen])  # all serialisable


async def test_budget_stop_is_reported(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartrix.core.session.build_agent", _scripted_agent(["python x.py", "python x.py"]))
    spec = _spec(ws, "task: go\npermissions: auto\nbudget: {max_tool_calls: 1}\napprovals: {allow: ['python *']}\n")
    outcome = await runner._run(spec)
    assert outcome.status == "stopped" and "tool-call budget" in (outcome.detail or "")
    assert runner.EXIT_CODES[outcome.status] == 2


def _plan() -> ExecutionPlan:
    task = PlannedTask(
        id="task_001", title="Write hello.py", description="Create hello.py", task_type=TaskType.IMPLEMENT,
        depends_on=[], estimated_minutes=5, output_files=["hello.py"], acceptance_criteria=["hello.py exists"],
    )  # fmt: skip
    return ExecutionPlan(
        project_name="hello", goal_summary="say hello", tech_stack=["python"], total_estimated_hours=0.1,
        tasks=[task], risks=[], assumptions=[],
    )  # fmt: skip


async def test_plan_only_is_auto_approved_recorded_and_resumable(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    planned: list[str] = []

    def fake_create_plan(goal: str, extra: str = "") -> ExecutionPlan:
        planned.append(goal)
        return _plan()

    monkeypatch.setattr("kartrix.tasks.orchestrator.create_plan", fake_create_plan)
    spec = _spec(ws, "task: build hello\nmode: plan\nplan_only: true\n")
    with collecting() as seen:
        outcome = await runner._run(spec)
    assert outcome.status == "completed" and outcome.plan is not None
    assert outcome.plan.status == "planned" and outcome.plan.plan["project_name"] == "hello"
    assert [t["status"] for t in outcome.plan.tasks] == ["pending"]
    reviewed = [e for e in seen if e.type == "plan_reviewed"]
    assert len(reviewed) == 1 and reviewed[0].approved and reviewed[0].by == "policy"

    report = runner.build_report(spec, "spec.yaml", outcome, seen, 0.0)
    assert report["plan"]["tasks"][0]["id"] == "task_001" and report["exit_code"] == 0

    again = await runner._run(spec)  # same goal: the saved project is resumed, not planned again
    assert again.plan is not None and again.plan.resumed and again.plan.project_id == outcome.plan.project_id
    assert planned == ["build hello"]
    other = await runner._run(_spec(ws, "task: something else\nmode: plan\nplan_only: true\n"))
    assert other.plan is not None and not other.plan.resumed  # a different goal never resumes it


def test_cli_writes_report_and_events(ws: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real entry point (its own event loop), with the run replaced: report, events file, exit code."""
    from kartrix.cli import main
    from kartrix.core.events import Usage, notice
    from kartrix.core.session import RunOutcome

    async def fake_run(spec: Any) -> RunOutcome:
        notice("working")
        return RunOutcome("ask", "failed", Usage(), answer="no", detail="tests failed")

    monkeypatch.setattr(runner, "_run", fake_run)
    (tmp_path / "s.yaml").write_text("task: x\n", encoding="utf-8")
    code = main(["run", "--spec", str(tmp_path / "s.yaml"), "--report", str(tmp_path / "r.json"),
                 "--events", str(tmp_path / "e.jsonl"), "--quiet"])  # fmt: skip
    assert code == 1
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert (report["status"], report["detail"], report["answer"]) == ("failed", "tests failed", "no")
    lines = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines == [{**lines[0], "type": "notice", "text": "working"}]
