"""Model fallback chain with circuit breaker (kartrix.llm.fallback) through a real ``create_agent`` loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import Field

from kartrix.llm import fallback
from kartrix.llm.factory import LazyModelFallbackMiddleware
from kartrix.llm.fallback import ModelUnavailableError


class _Status(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"[{status}] provider error")
        self.status_code = status


class _Model(GenericFakeChatModel):
    model: str
    fail_with: list[BaseException | None] = Field(default_factory=list)  # one per call; None = answer
    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _Model:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        error = self.fail_with.pop(0) if self.fail_with else None
        if error is not None:
            raise error
        return super()._generate(messages, *args, **kwargs)


def _model(name: str, *fail_with: BaseException | None) -> _Model:
    return _Model(model=name, messages=iter([AIMessage(f"answer from {name}")] * 5), fail_with=list(fail_with))


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "RETRY_PAUSE", 0.0)
    fallback.reset_circuit()
    yield
    fallback.reset_circuit()


def _agent(primary: _Model, *fallbacks: _Model) -> Any:
    return create_agent(primary, tools=[], middleware=[LazyModelFallbackMiddleware([lambda m=m: m for m in fallbacks])])


async def _ask(agent: Any) -> str:
    out = await agent.ainvoke({"messages": [HumanMessage("hi")]})
    return str(out["messages"][-1].content)


async def test_falls_back_and_skips_a_provider_without_credits() -> None:
    primary, backup = _model("primary", _Status(402)), _model("backup")
    agent = _agent(primary, backup)
    assert await _ask(agent) == "answer from backup"
    assert await _ask(agent) == "answer from backup"
    assert primary.calls == 1  # the second call didn't try the primary again


async def test_timed_out_primary_gets_a_second_chance() -> None:
    primary, backup = _model("primary", TimeoutError("read timed out")), _model("backup", _Status(404))
    assert await _ask(_agent(primary, backup)) == "answer from primary"
    assert primary.calls == 2 and backup.calls == 1


async def test_everything_failing_gives_one_clear_error() -> None:
    primary = _model("primary", TimeoutError(), TimeoutError())
    backup = _model("backup", _Status(402))
    with pytest.raises(ModelUnavailableError) as info:
        await _ask(_agent(primary, backup))
    message = str(info.value)
    assert "primary: timed out" in message and "backup: out of credits (402)" in message
    assert "second try" in message


async def test_success_closes_the_circuit_for_a_timed_out_model() -> None:
    primary, backup = _model("primary", TimeoutError()), _model("backup")
    agent = _agent(primary, backup)
    assert await _ask(agent) == "answer from backup"
    assert "primary" in fallback._circuit  # cooling down after the timeout
    fallback._circuit["primary"].until = 0  # cool-down over
    assert await _ask(agent) == "answer from primary"
    assert "primary" not in fallback._circuit


def test_fallback_clients_are_not_built_when_the_primary_answers() -> None:
    built: list[str] = []

    def factory() -> _Model:
        built.append("backup")
        return _model("backup")

    agent = create_agent(_model("primary"), tools=[], middleware=[LazyModelFallbackMiddleware([factory])])
    agent.invoke({"messages": [HumanMessage("hi")]})
    assert built == []
