"""Budgets and the kill switch (B9)."""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from kartrix.config import BudgetLimits, ModelPrice
from kartrix.security.audit import classify_outcome
from kartrix.security.budget import (
    Budget,
    BudgetMiddleware,
    RunStopped,
    budget_scope,
    command_stop_check,
    raise_if_stopped,
    waiting_for_user,
)
from kartrix.security.kill_switch import (
    CTRL_C_REASON,
    STOP_COMMAND_REASON,
    close_dangling_tool_calls,
    request_stop,
    run_stoppable,
)
from kartrix.tools.process_runner import ProcessResult, run_process
from kartrix.tools.terminal_tools import format_result


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))  # the stop file lives here


def _budget(**limits: Any) -> Budget:
    return Budget("turn", BudgetLimits(**limits), {"priced": ModelPrice(input=3.0, output=15.0)})


# ── accounting and limits ─────────────────────────────────────────────


def test_tokens_and_cost_are_counted_and_limited() -> None:
    b = _budget(max_tokens=10_000, max_cost_usd=1.0)
    b.add_model_call("priced", 1000, 200)
    assert b.total_tokens == 1200
    assert b.cost_usd == pytest.approx((1000 * 3 + 200 * 15) / 1_000_000)
    assert b.check() is None
    b.add_model_call("priced", 8000, 800)
    assert "token budget reached" in (b.check() or "")
    assert not b.hard  # a token limit stops at the next step; it doesn't kill commands


def test_cost_limit_and_unpriced_models() -> None:
    b = _budget(max_cost_usd=0.01)
    b.add_model_call("free-model", 50_000, 50_000)  # no price: counted, not charged
    assert b.check() is None and b.unpriced == {"free-model"}
    assert "cost unknown for free-model" in b.summary()
    b.add_model_call("priced", 0, 1000)  # $0.015
    assert "cost budget reached" in (b.check() or "")


def test_time_limit_is_hard_and_excludes_waiting_for_the_user() -> None:
    b = _budget(max_seconds=0.3)
    with budget_scope(b), waiting_for_user():
        time.sleep(0.4)
    assert b.check() is None  # the wait didn't count
    time.sleep(0.35)
    assert "time budget reached" in (b.check() or "")
    assert b.hard and b.hard_stop_reason()


def test_kartrix_stop_stops_runs_started_before_it_only() -> None:
    running = _budget()
    request_stop()
    assert running.check() == STOP_COMMAND_REASON and running.hard
    later = _budget()
    assert later.check() is None  # started after the stop request: unaffected


def test_first_stop_reason_wins_and_raise_if_stopped() -> None:
    b = _budget()
    with budget_scope(b):
        raise_if_stopped()  # nothing yet
        b.stop("first")
        b.stop("second", hard=True)
        with pytest.raises(RunStopped, match="first"):
            raise_if_stopped()
    assert b.hard
    raise_if_stopped()  # outside a run: no-op


def test_command_stop_check_reports_only_hard_stops() -> None:
    assert command_stop_check() is None
    b = _budget()
    with budget_scope(b):
        check = command_stop_check()
        assert check is not None
        b.stop("token budget reached")
        assert check() is None  # soft: let the command finish
        b.stop("x", hard=True)
        assert check() == "token budget reached"


# ── middleware through a real agent loop ──────────────────────────────


class _FakeModel(GenericFakeChatModel):
    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return super()._generate(*args, **kwargs)


def _call(i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": f"c{i}"}])


ran: list[int] = []


@tool
def ping() -> str:
    """Ping."""
    ran.append(1)
    return "pong"


def _agent(script: list[AIMessage]) -> tuple[Any, _FakeModel]:
    model = _FakeModel(messages=iter(script))
    return create_agent(model, tools=[ping], middleware=[BudgetMiddleware()]), model


async def _run(agent: Any) -> list[Any]:
    result = await agent.ainvoke({"messages": [HumanMessage("go")]})
    return result["messages"]


async def test_tool_call_limit_refuses_then_allows_one_wrap_up_then_stops() -> None:
    ran.clear()
    agent, model = _agent([_call(1), _call(2), _call(3), _call(4), AIMessage("never reached")])
    b = _budget(max_tool_calls=2)
    with budget_scope(b):
        messages = await _run(agent)
    assert len(ran) == 2 and b.tool_calls == 2
    refusals = [m for m in messages if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(refusals) == 2 and "Don't call more tools" in refusals[0].content
    assert model.calls == 4  # the 4th call was the wrap-up chance; the 5th never happened
    assert messages[-1].content.startswith("Stopped: tool-call budget reached (2 calls)")
    assert b.exhausted()


async def test_wrap_up_answer_ends_the_run_normally() -> None:
    ran.clear()
    agent, _ = _agent([_call(1), _call(2), AIMessage("did 1 of 2, ping again later")])
    b = _budget(max_tool_calls=1)
    with budget_scope(b):
        messages = await _run(agent)
    assert messages[-1].content == "did 1 of 2, ping again later"
    assert b.stop_reason is None
    assert "tool-call budget" in (b.exhausted() or "")  # a task would not be judged as done


async def test_token_limit_stops_before_the_next_model_call() -> None:
    first = _call(1)
    first.usage_metadata = {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110}
    agent, model = _agent([first, AIMessage("never reached")])
    b = _budget(max_tokens=100)
    with budget_scope(b):
        messages = await _run(agent)
    assert model.calls == 1 and (b.input_tokens, b.output_tokens) == (80, 30) and not b.estimated
    assert messages[-1].content.startswith("Stopped: token budget reached")
    assert isinstance(messages[-2], ToolMessage)  # the tool call of the paid-for turn still got its result


async def test_tokens_are_estimated_when_the_provider_sends_no_usage() -> None:
    agent, _ = _agent([AIMessage("hello there")])
    b = _budget()
    with budget_scope(b):
        await _run(agent)
    assert b.estimated and b.input_tokens > 0 and b.output_tokens > 0 and b.model_calls == 1
    assert b.summary().startswith("~")


async def test_stopped_run_refuses_tools_and_ends() -> None:
    ran.clear()
    agent, _ = _agent([_call(1), AIMessage("never")])
    b = _budget()
    with budget_scope(b):
        b.stop(CTRL_C_REASON, hard=True)
        messages = await _run(agent)
    assert ran == [] and messages[-1].content.startswith(f"Stopped: {CTRL_C_REASON}")


async def test_middleware_is_a_no_op_outside_a_run() -> None:
    ran.clear()
    agent, _ = _agent([_call(1), AIMessage("done")])
    messages = await _run(agent)
    assert ran == [1] and messages[-1].content == "done"


# ── kill switch ───────────────────────────────────────────────────────


async def test_kartrix_stop_cancels_a_run_in_flight() -> None:
    b = _budget()
    asyncio.get_running_loop().call_later(0.1, request_stop)
    start = time.monotonic()
    assert await run_stoppable(asyncio.sleep(10, result="finished"), b) is None
    assert time.monotonic() - start < 3
    assert b.stop_reason == STOP_COMMAND_REASON


async def test_run_stoppable_returns_the_result_and_sees_the_budget() -> None:
    b = _budget()

    async def work() -> bool:
        await asyncio.sleep(0)
        return command_stop_check() is not None  # the budget is visible inside the task

    assert await run_stoppable(work(), b) is True
    assert b.stop_reason is None


@pytest.mark.skipif(not hasattr(signal, "raise_signal"), reason="needs signal.raise_signal")
async def test_ctrl_c_stops_the_run_but_not_kartrix() -> None:
    b = _budget()
    messages: list[str] = []
    before = signal.getsignal(signal.SIGINT)
    asyncio.get_running_loop().call_later(0.1, signal.raise_signal, signal.SIGINT)
    assert await run_stoppable(asyncio.sleep(10), b, on_stop=messages.append) is None
    assert b.stop_reason == CTRL_C_REASON and messages == ["Stopping…"]
    assert signal.getsignal(signal.SIGINT) == before  # the previous handler is back


@pytest.mark.skipif(not hasattr(signal, "raise_signal"), reason="needs signal.raise_signal")
async def test_ctrl_c_while_waiting_for_the_user_is_not_cancelled() -> None:
    b = _budget()
    messages: list[str] = []

    async def ask() -> str:
        with waiting_for_user():
            await asyncio.sleep(0.4)  # stands in for the approval prompt
        return "answered"

    asyncio.get_running_loop().call_later(0.1, signal.raise_signal, signal.SIGINT)
    assert await run_stoppable(ask(), b, on_stop=messages.append) == "answered"
    assert b.stop_reason == CTRL_C_REASON and "counts as 'no'" in messages[0]


async def test_dangling_tool_calls_are_answered_after_a_cancelled_turn() -> None:
    model = _FakeModel(messages=iter([AIMessage("ok")]))
    agent = create_agent(model, tools=[ping], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    # State as a cancellation leaves it: the model asked for a tool, the tool never answered.
    await agent.aupdate_state(config, {"messages": [HumanMessage("go"), _call(1)]}, as_node="model")
    assert await close_dangling_tool_calls(agent, config, CTRL_C_REASON) == 1
    messages = (await agent.aget_state(config)).values["messages"]
    assert isinstance(messages[-2], ToolMessage) and messages[-2].tool_call_id == "c1"
    assert messages[-1].content == f"Stopped: {CTRL_C_REASON}."
    assert await close_dangling_tool_calls(agent, config, CTRL_C_REASON) == 0  # idempotent
    result = await agent.ainvoke({"messages": [HumanMessage("again")]}, config)
    assert result["messages"][-1].content == "ok"


def test_should_stop_kills_a_running_command() -> None:
    start = time.monotonic()
    threshold = start + 0.5

    def should_stop() -> str | None:
        return "stopped by test" if time.monotonic() > threshold else None

    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], None, {}, 60, should_stop)
    assert result.stopped == "stopped by test" and not result.timed_out
    assert time.monotonic() - start < 5


def test_stopped_command_output_and_audit_outcome() -> None:
    text = format_result(ProcessResult(None, "partial", "", False, CTRL_C_REASON), 300)
    assert "Error: stopped" in text and "killed" in text
    assert classify_outcome("Error: stopped — token budget reached. The tool was not run.") == "stopped"
