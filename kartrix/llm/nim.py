"""NVIDIA NIM chat model with tool calls recovered from text.

Asked for a forced tool call (``tool_choice="required"``, which LangChain's ``ToolStrategy`` structured output
sends as ``"any"``), Nemotron 3 Ultra on NIM sometimes writes the calls into the message text instead —
``[[{"name": "glob", "parameters": {"pattern": "**/test*.py"}}]]``, brackets not always balanced — and stops
with ``finish_reason: stop`` and no tool calls. The agent loop then ends: the reviewer finished "without a
verdict" although it was only looking for the tests. ``ChatNVIDIA`` here turns such a message back into the
tool calls it spells out, when the text is nothing but calls to tools that were bound.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA as _BaseChatNVIDIA

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_WRAPPER = re.compile(r"</?TOOLCALL>|</?tool_call>")  # Nemotron's own tool-call tags
_SEPARATORS = frozenset("[],")
_decoder = json.JSONDecoder()


def _args(call: dict[str, Any]) -> dict[str, Any] | None:
    args = call.get("parameters", call.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return args if isinstance(args, dict) else None


def parse_text_tool_calls(text: str, tool_names: set[str] | None) -> list[ToolCall] | None:
    """The tool calls ``text`` consists of (JSON objects with ``name`` and ``parameters``/``arguments``, in
    brackets or not), or None when it holds anything else or names a tool that wasn't bound (``tool_names``;
    None: any name)."""
    text = _WRAPPER.sub("", text)
    calls: list[ToolCall] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in _SEPARATORS:
            i += 1
            continue
        if ch != "{":
            return None
        try:
            obj, i = _decoder.raw_decode(text, i)
        except ValueError:
            return None
        name = obj.get("name") if isinstance(obj, dict) else None
        args = _args(obj) if isinstance(name, str) and (tool_names is None or name in tool_names) else None
        if name is None or args is None:
            return None
        calls.append(ToolCall(name=name, args=args, id=f"call-{uuid.uuid4()}", type="tool_call"))
    return calls or None


def _bound_tool_names(kwargs: dict[str, Any]) -> set[str]:
    names = set()
    for tool in kwargs.get("tools") or []:
        name = tool.get("function", {}).get("name") if isinstance(tool, dict) else None
        if name:
            names.add(name)
    return names


def recover_tool_calls(message: AIMessage, tool_names: set[str]) -> AIMessage:
    """``message`` with the tool calls its text spells out as real tool calls (unchanged if there are none)."""
    if message.tool_calls or not tool_names or not isinstance(message.content, str):
        return message
    calls = parse_text_tool_calls(message.content, tool_names)
    if calls is None:
        return message
    logger.info("Recovered tool calls written as text", extra={"tools": [c["name"] for c in calls]})
    return message.model_copy(update={"content": "", "tool_calls": calls, "invalid_tool_calls": []})


class RecoverTextToolCalls(BaseChatModel):
    """A chat model whose replies get tool calls written as text recovered — put it before the model class.
    ``invoke`` sees the bound tools (``bind_tools`` passes them as ``tools``) and the finished message
    whether the call streamed or not."""

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, *, stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:  # fmt: skip
        message = super().invoke(input, config, stop=stop, **kwargs)
        return recover_tool_calls(message, _bound_tool_names(kwargs))

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, *, stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:  # fmt: skip
        message = await super().ainvoke(input, config, stop=stop, **kwargs)
        return recover_tool_calls(message, _bound_tool_names(kwargs))


class ChatNVIDIA(RecoverTextToolCalls, _BaseChatNVIDIA):
    """``ChatNVIDIA`` whose replies get tool calls written as text recovered (module docstring)."""
