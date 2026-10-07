"""The multi-agent chat graph (kartrix.agent.graph) with scripted models: routing, the explorer → coder ⇄
reviewer loop, an approval raised inside a subagent pausing and resuming the whole graph, and budgets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from kartrix.agent import graph as graph_mod
from kartrix.agent.factory import agent_middleware
from kartrix.config import BudgetLimits, settings
from kartrix.core.events import collecting
from kartrix.sandbox.manager import set_sandbox
from kartrix.security import permissions
from kartrix.security import workspace as ws_mod
from kartrix.security.approvals import ApprovalDecision, ApprovalMiddleware, run_agent
from kartrix.security.budget import Budget, budget_scope

pytestmark = pytest.mark.usefixtures("db", "fake_embedder")


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self


def _call(name: str, args: dict[str, Any], i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}_{i}"}])


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "app.py").write_text("def add(a, b):\n    return a - b\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings.llm, "fallbacks", [])
    ws_mod.set_workspace(root)
    set_sandbox(None)
    yield root
    ws_mod._current = None
    permissions._mode = None
    permissions.clear_session_allowances()


def _graph(monkeypatch: pytest.MonkeyPatch, router: list[AIMessage], main: list[AIMessage]) -> Any:
    models = {"router": _FakeModel(messages=iter(router)), "main": _FakeModel(messages=iter(main))}
    monkeypatch.setattr(graph_mod, "get_chat_model", lambda role="main", **kw: models[role])
    built = graph_mod.AgentGraph(middleware=agent_middleware, approval=ApprovalMiddleware)
    return built.compile(InMemorySaver())


async def _approve_all(requests: list[Any]) -> list[ApprovalDecision]:
    return [ApprovalDecision("approve", by="user") for _ in requests]


async def _ask(agent: Any, text: str, thread: str = "t1", approver: Any = _approve_all) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    with budget_scope(Budget("turn", BudgetLimits(max_tool_calls=50))):
        return await run_agent(agent, {"messages": [{"role": "user", "content": text}]}, config, approver)


async def test_question_goes_through_the_explorer(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _graph(
        monkeypatch,
        router=[_call("RouteDecision", {"route": "question", "reason": "asks about code"})],
        main=[
            _call("read_file", {"file_path": "app.py"}),
            AIMessage("add() in app.py subtracts instead of adding (line 2)."),
            AIMessage("`add` in app.py:2 returns a - b, so it subtracts."),
        ],
    )
    with collecting() as seen:
        state = await _ask(agent, "What does add do?")
    assert state["messages"][-1].content == "`add` in app.py:2 returns a - b, so it subtracts."
    assert state["steps"] == ["router", "explorer", "responder"]
    assert "subtracts" in state["findings"]
    steps = [(e.agent, e.status) for e in seen if e.type == "agent_step"]
    assert ("explorer", "started") in steps and ("coder", "started") not in steps
    calls = [e for e in seen if e.type == "tool_call_started"]
    assert [c.tool for c in calls] == ["read_file"]  # the explorer's tool calls are audited like any other


async def test_chat_skips_the_subagents(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _graph(
        monkeypatch,
        router=[_call("RouteDecision", {"route": "chat", "reason": "greeting"})],
        main=[AIMessage("Hi! Ask me anything about this repository.")],
    )
    state = await _ask(agent, "hello")
    assert state["steps"] == ["router", "responder"]


async def test_change_loops_until_the_reviewer_approves(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = "def add(a, b):\n    return a + b\n"
    agent = _graph(
        monkeypatch,
        router=[_call("RouteDecision", {"route": "change", "reason": "fix"})],
        main=[
            AIMessage("add() in app.py subtracts."),  # explorer
            _call("write_file", {"file_path": "app.py", "content": fixed}, 1),  # coder, round 1
            AIMessage("Fixed add in app.py."),
            _call("ReviewVerdict", {"approved": False, "issues": ["no test for add"]}, 2),  # reviewer
            _call("write_file", {"file_path": "test_app.py", "content": "from app import add\n"}, 3),  # coder, round 2
            AIMessage("Added test_app.py."),
            _call("ReviewVerdict", {"approved": True, "issues": []}, 4),  # reviewer
            AIMessage("Fixed `add` and added a test; the reviewer approved."),  # responder
        ],
    )
    state = await _ask(agent, "Fix add so it adds")
    assert (ws / "app.py").read_text() == fixed and (ws / "test_app.py").is_file()
    assert state["steps"] == ["router", "explorer", "coder", "reviewer", "coder", "reviewer", "responder"]
    assert state["approved"] is True and state["rounds"] == 2


async def test_approval_inside_a_subagent_pauses_and_resumes_the_graph(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ws / "x.py").write_text("import pathlib; pathlib.Path('ran.txt').write_text('yes')\n")
    agent = _graph(
        monkeypatch,
        router=[_call("RouteDecision", {"route": "change", "reason": "run"})],
        main=[
            AIMessage("x.py writes ran.txt."),
            _call("run_command", {"command": "python x.py"}, 1),  # needs approval: no sandbox
            AIMessage("Ran x.py."),
            _call("ReviewVerdict", {"approved": True, "issues": []}, 2),
            AIMessage("Done: ran x.py."),
        ],
    )
    asked: list[str] = []

    async def approver(requests: list[Any]) -> list[ApprovalDecision]:
        asked.extend(r.command for r in requests)
        return [ApprovalDecision("approve", by="user") for _ in requests]

    state = await _ask(agent, "Run x.py", approver=approver)
    assert asked == ["python x.py"]
    assert (ws / "ran.txt").read_text() == "yes"
    assert state["messages"][-1].content == "Done: ran x.py."


async def test_reached_budget_skips_to_the_answer(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _graph(monkeypatch, router=[_call("RouteDecision", {"route": "question", "reason": "q"})], main=[])
    config = {"configurable": {"thread_id": "t2"}}
    budget = Budget("turn", BudgetLimits(max_tool_calls=10))
    budget.stop("token budget reached")
    with budget_scope(budget):
        state = await run_agent(agent, {"messages": [{"role": "user", "content": "q"}]}, config, _approve_all)
    assert state["messages"][-1].content.startswith("Stopped:")
    assert "explorer" not in state.get("steps", [])
