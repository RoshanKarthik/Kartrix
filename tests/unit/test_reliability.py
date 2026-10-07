"""The completion guard: empty / cut-off model turns, unparseable tool calls and crashing tools
(kartrix.agent.reliability) through a real ``create_agent`` loop with a scripted model."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import Field

from kartrix.agent.reliability import (
    EMPTY_NUDGE,
    LENGTH_EMPTY_NUDGE,
    LENGTH_PARTIAL_NUDGE,
    CompletionGuardMiddleware,
    final_text,
    is_nudge,
)


class _Model(GenericFakeChatModel):
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> _Model:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _model(*replies: AIMessage) -> _Model:
    return _Model(messages=iter(replies), seen=[])


@tool
def ping() -> str:
    """Answer pong."""
    return "pong"


@tool
def explode() -> str:
    """Always fails."""
    raise RuntimeError("disk on fire")


def _agent(model: _Model, max_nudges: int = 2) -> Any:
    return create_agent(model, tools=[ping, explode], middleware=[CompletionGuardMiddleware(max_nudges=max_nudges)])


def _cut(content: str = "") -> AIMessage:
    return AIMessage(content=content, response_metadata={"finish_reason": "length"})


async def test_empty_turn_is_nudged_and_dropped() -> None:
    model = _model(AIMessage(content=""), AIMessage(content="All done."))
    out = await _agent(model).ainvoke({"messages": [HumanMessage("do it")]})
    assert final_text(out["messages"]) == "All done."
    second_call = model.seen[1]
    assert is_nudge(second_call[-1]) and second_call[-1].content == EMPTY_NUDGE
    assert not any(isinstance(m, AIMessage) and m.content == "" for m in out["messages"])  # empty turn removed


async def test_thinking_cut_off_by_the_output_limit_is_nudged() -> None:
    model = _model(
        _cut(), AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": "c1"}]), AIMessage("ok")
    )
    out = await _agent(model).ainvoke({"messages": [HumanMessage("do it")]})
    assert model.seen[1][-1].content == LENGTH_EMPTY_NUDGE
    assert any(isinstance(m, ToolMessage) and m.content == "pong" for m in out["messages"])
    assert final_text(out["messages"]) == "ok"


async def test_answer_cut_off_mid_text_is_continued_and_joined() -> None:
    model = _model(_cut("The fix is in slug"), AIMessage(content="ify.ts, line 4."))
    out = await _agent(model).ainvoke({"messages": [HumanMessage("where?")]})
    assert model.seen[1][-1].content == LENGTH_PARTIAL_NUDGE
    assert final_text(out["messages"]) == "The fix is in slugify.ts, line 4."


async def test_unparseable_tool_call_is_reported_back() -> None:
    bad = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "ping", "args": "{oops", "id": "c1", "error": "bad JSON", "type": "invalid_tool_call"}
        ],
    )
    model = _model(bad, AIMessage(content="", tool_calls=[{"name": "ping", "args": {}, "id": "c2"}]), AIMessage("ok"))
    out = await _agent(model).ainvoke({"messages": [HumanMessage("ping")]})
    nudge = model.seen[1][-1]
    assert is_nudge(nudge) and "could not be parsed" in str(nudge.content) and "bad JSON" in str(nudge.content)
    assert final_text(out["messages"]) == "ok"


async def test_gives_up_after_max_nudges() -> None:
    model = _model(AIMessage(""), AIMessage(""), AIMessage(""), AIMessage("never reached"))
    out = await _agent(model, max_nudges=2).ainvoke({"messages": [HumanMessage("do it")]})
    assert len(model.seen) == 3  # first try + 2 nudges
    assert final_text(out["messages"]) == ""


async def test_nudges_are_counted_per_request() -> None:
    model = _model(AIMessage(""), AIMessage("first"), AIMessage(""), AIMessage("second"))
    agent = _agent(model, max_nudges=1)
    out = await agent.ainvoke({"messages": [HumanMessage("one")]})
    out = await agent.ainvoke({"messages": [*out["messages"], HumanMessage("two")]})
    assert final_text(out["messages"]) == "second"


async def test_crashing_tool_becomes_an_error_result() -> None:
    model = _model(AIMessage(content="", tool_calls=[{"name": "explode", "args": {}, "id": "c1"}]), AIMessage("sorry"))
    out = await _agent(model).ainvoke({"messages": [HumanMessage("go")]})
    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert result.status == "error" and "RuntimeError: disk on fire" in str(result.content)
    assert final_text(out["messages"]) == "sorry"


def test_final_text_stops_at_tool_activity() -> None:
    msgs = [
        HumanMessage("q"),
        AIMessage("early text", tool_calls=[{"name": "ping", "args": {}, "id": "c"}]),
        ToolMessage("pong", tool_call_id="c"),
        AIMessage("answer"),
    ]
    assert final_text(msgs) == "answer"
    assert final_text([]) == ""
