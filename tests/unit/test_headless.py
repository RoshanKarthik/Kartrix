"""Headless mode (K1) without services: spec loading, the policy approver, budgets, the report, exit codes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kartrix.config import settings
from kartrix.core import events as ev
from kartrix.headless.policy import PolicyApprover
from kartrix.headless.runner import EXIT_CODES, _limits, build_report, run_headless
from kartrix.headless.spec import ApprovalPolicy, SpecError, load_spec
from kartrix.security.approvals import ApprovalRequest


def _spec(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_spec_defaults(tmp_path: Path) -> None:
    spec = load_spec(_spec(tmp_path, "task: '  explain main.py  '\n"))
    assert (spec.task, spec.mode, spec.permissions, spec.plan_only, spec.semantic_cache) == (
        "explain main.py", "ask", "default", False, False
    )  # fmt: skip
    assert spec.approvals.otherwise == "reject" and spec.approvals.allow == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("mode: ask\n", "task"),
        ("task: ''\n", "must not be empty"),
        ("task: x\ntask: y\n", "uplicate"),
        ("task: x\nmodel: gpt\n", "Extra inputs"),
        ("task: x\napprovals: {allow: [a], otherwise: approve}\n", "otherwise"),
        ("task: x\nmode: build\n", "mode"),
        ("- task\n", "mapping"),
    ],
)
def test_invalid_specs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        load_spec(_spec(tmp_path, text))


def test_missing_spec_exits_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_headless(str(tmp_path / "nope.yaml")) == EXIT_CODES["not_started"] == 3
    assert "spec file not found" in capsys.readouterr().err


def test_budget_overrides_only_the_given_fields(tmp_path: Path) -> None:
    spec = load_spec(_spec(tmp_path, "task: x\nmode: plan\nbudget: {max_cost_usd: 0.5, max_seconds: null}\n"))
    limits = _limits(spec)
    assert limits.max_cost_usd == 0.5 and limits.max_seconds is None  # explicit null = no limit
    assert limits.max_tokens == settings.budgets.plan.max_tokens
    assert _limits(load_spec(_spec(tmp_path, "task: x\n"))) == settings.budgets.turn


def _req(command: str, tool: str = "run_command") -> ApprovalRequest:
    return ApprovalRequest("id", tool, command, ".", "network", "why", "default", True)


async def test_policy_approver() -> None:
    approver = PolicyApprover(
        ApprovalPolicy(allow=["npm install *", "pytest *", "curl *"], deny=["curl *"], allow_tools=["create_issue"])
    )
    got = await approver([_req("npm install left-pad"), _req("pytest -q"), _req("curl https://x"), _req("rm -rf build"),
                          _req("create_issue t", "create_issue"), _req("delete_repo r", "delete_repo"),
                          _req("npm install 'unterminated")])  # fmt: skip
    assert [d.type for d in got] == ["approve", "approve", "reject", "reject", "approve", "reject", "reject"]
    assert all(d.by == "policy" for d in got)
    assert "denied by this run's approval policy" in (got[2].message or "")  # deny beats allow
    assert "no approvals.allow rule matches" in (got[3].message or "")


def test_report_from_events(tmp_path: Path) -> None:
    spec = load_spec(_spec(tmp_path, "task: go\npermissions: auto\n"))
    events: list[ev.Event] = [
        ev.ToolCallFinished(call_id="1", tool="run_command", outcome="ok", duration_ms=1),
        ev.ToolCallFinished(call_id="2", tool="run_command", outcome="declined", duration_ms=1),
        ev.ToolCallFinished(call_id="3", tool="read_file", outcome="ok", duration_ms=1),
        ev.ApprovalRequested(
            tool_call_id="2",
            tool="run_command",
            command="curl x",
            directory=".",
            category="network",
            reason="other hosts",
        ),
        ev.ApprovalResolved(tool_call_id="2", decision="reject", by="policy", message="no"),
        ev.FilesChanged(label="/ask go", files=["b.py", "a.py"], count=2),
        ev.FilesChanged(label="task t", files=["a.py"], count=1),
    ]
    outcome = SimpleNamespace(
        status="stopped", detail="budget", usage=ev.Usage(tool_calls=3), answer="partial", cached=False, plan=None
    )
    report = build_report(spec, "spec.yaml", outcome, events, 0.0)
    assert (report["status"], report["exit_code"], report["detail"]) == ("stopped", 2, "budget")
    assert report["tool_calls"] == {
        "total": 3, "by_tool": {"run_command": 2, "read_file": 1}, "by_outcome": {"ok": 2, "declined": 1}
    }  # fmt: skip
    assert report["approvals"] == [
        {"tool": "run_command", "command": "curl x", "category": "network", "reason": "other hosts",
         "task_key": None, "decision": "reject", "by": "policy", "message": "no"}
    ]  # fmt: skip
    assert report["files_changed"] == ["a.py", "b.py"] and report["usage"]["tool_calls"] == 3
    assert report["permissions"] == "auto" and report["plan"] is None
