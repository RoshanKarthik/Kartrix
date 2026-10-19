"""Structured output strategy (kartrix.llm.factory.response_format_for) through a real ``create_agent`` loop.

The fake model behaves like gpt-oss on NVIDIA NIM with ``langchain-nvidia-ai-endpoints`` 1.5: its profile claims
native structured output, but asked for JSON natively it answers with an empty message; a forced tool call works.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from kartrix.agent.graph import ROUTER_PROMPT, RouteDecision
from kartrix.llm.factory import response_format_for


class _GptOssLike(GenericFakeChatModel):
    model: str = "openai/gpt-oss-20b"
    native: bool = False  # bound with a native response_format (ProviderStrategy)
    tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> _GptOssLike:
        names = [t.name if hasattr(t, "name") else t["function"]["name"] for t in tools]
        return self.model_copy(update={"native": "response_format" in kwargs, "tool_names": names})

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if self.native:
            reply = AIMessage(content="", additional_kwargs={"reasoning_content": 'Inspect repo.\n{"path": ""}'})
        else:
            args_ = {"route": "change", "reason": "asks for a new flag"}
            reply = AIMessage(content="", tool_calls=[{"name": self.tool_names[0], "args": args_, "id": "call_1"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _model() -> _GptOssLike:
    return _GptOssLike(messages=iter([]), profile={"structured_output": True, "tool_calling": True})


async def _route(response_format: Any) -> Any:
    agent = create_agent(_model(), tools=[], system_prompt=ROUTER_PROMPT, response_format=response_format)
    out = await agent.ainvoke({"messages": [HumanMessage("Add a --json flag to the stats command")]})
    return out["structured_response"]


async def test_bare_schema_picks_native_output_from_the_profile_and_fails() -> None:
    """The failure seen on a fresh install: LangChain trusts the profile, the model answers with nothing."""
    with pytest.raises(StructuredOutputValidationError, match="Native structured output"):
        await _route(RouteDecision)


async def test_router_format_forces_a_tool_call_on_nim() -> None:
    decision = await _route(response_format_for(RouteDecision, "router"))
    assert decision == RouteDecision(route="change", reason="asks for a new flag")
