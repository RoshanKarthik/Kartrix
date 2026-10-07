"""Keeping agent loops alive when the model or a tool misbehaves.

Reasoning models (Nemotron, gpt-oss) think before they answer. A turn whose thinking fills the
output limit comes back with ``finish_reason: length`` and nothing else — no text, no tool call —
and a plain ReAct loop treats that as "done", ending the run with an empty answer. The same loop
also ends on a tool call whose JSON arguments couldn't be parsed, and a tool that raises an
unexpected exception aborts the whole run.

``CompletionGuardMiddleware`` turns those into recoverable steps:

- an empty or cut-off turn without tool calls → the empty message is dropped and the model is
  nudged to continue (at most ``agents.max_nudges`` times per request);
- unparseable tool calls → the model is told what failed and asked to call the tool again;
- a tool that raises → an ``Error: ...`` tool result the model can react to, instead of a crash.

Interrupts (approvals), cancellation (the kill switch) and budget stops pass through untouched.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.security.budget import RunStopped
from kartrix.security.budget import current as current_budget

logger = get_logger(__name__)

NUDGE_NAME = "kartrix_nudge"  # name of the HumanMessages this middleware adds

EMPTY_NUDGE = (
    "You ended your turn without a tool call or an answer. Continue the task: call the next tool you "
    "need, or, if you are done, write your final report now."
)
LENGTH_EMPTY_NUDGE = (
    "Your last reply hit the output limit while you were still thinking, so nothing was sent. Think less "
    "and act: call the next tool you need, or write your final report now."
)
LENGTH_PARTIAL_NUDGE = "Your reply was cut off by the output limit. Continue exactly where you stopped."


def text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def is_nudge(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) and message.name == NUDGE_NAME


def _finish_reason(message: AIMessage) -> str:
    meta = message.response_metadata or {}
    return str(meta.get("finish_reason") or meta.get("stop_reason") or "")


def _cut_off(message: AIMessage) -> bool:
    return _finish_reason(message) in ("length", "max_tokens")


def final_text(messages: list[AnyMessage]) -> str:
    """The final answer of an agent run: the text of the trailing assistant turns, so an answer that was
    cut off and continued after a nudge is returned whole."""
    parts: list[str] = []
    for m in reversed(messages):
        if is_nudge(m):
            continue
        if isinstance(m, AIMessage) and not m.tool_calls:
            parts.append(text_of(m.content))
            continue
        break
    return "".join(reversed(parts)).strip()


def _nudges_since_request(messages: list[AnyMessage]) -> int:
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if not is_nudge(m):
                break
            count += 1
    return count


def _nudge_for(message: AIMessage) -> str | None:
    """What to tell the model about a final turn that isn't a usable answer (None: it is fine)."""
    if message.tool_calls:
        return None
    if message.invalid_tool_calls:
        details = "; ".join(
            f"{c.get('name') or '?'}: {c.get('error') or 'invalid JSON arguments'}" for c in message.invalid_tool_calls
        )
        return (
            f"Your last tool call could not be parsed ({details[:500]}). Call the tool again with valid JSON "
            "arguments that match its schema."
        )
    text = text_of(message.content).strip()
    if _cut_off(message):
        return LENGTH_PARTIAL_NUDGE if text else LENGTH_EMPTY_NUDGE
    return None if text else EMPTY_NUDGE


class CompletionGuardMiddleware(AgentMiddleware):
    """Recover from empty/cut-off model turns, unparseable tool calls and crashing tools (module docstring)."""

    def __init__(self, max_nudges: int | None = None) -> None:
        super().__init__()
        self.max_nudges = settings.agents.max_nudges if max_nudges is None else max_nudges

    def _check(self, state: Any) -> dict[str, Any] | None:
        messages: list[AnyMessage] = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        last = messages[-1]
        nudge = _nudge_for(last)
        if nudge is None:
            return None
        budget = current_budget()
        if budget is not None and budget.exhausted():
            return None
        used = _nudges_since_request(messages)
        if used >= self.max_nudges:
            logger.warning(
                "Model kept ending turns without a usable answer; giving up",
                extra={"nudges": used, "finish_reason": _finish_reason(last)},
            )
            return None
        logger.info("Nudging the model to continue", extra={"nudge": used + 1, "finish_reason": _finish_reason(last)})
        update: list[AnyMessage | RemoveMessage] = []
        keep = bool(text_of(last.content).strip()) and nudge == LENGTH_PARTIAL_NUDGE
        if not keep and last.id:  # an empty turn only confuses the next call
            update.append(RemoveMessage(id=last.id))
        update.append(HumanMessage(nudge, name=NUDGE_NAME))
        return {"messages": update, "jump_to": "model"}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._check(state)

    @staticmethod
    def _error_result(request: ToolCallRequest, e: Exception) -> ToolMessage:
        call = request.tool_call
        logger.warning(
            "Tool raised; returned the error to the model", extra={"tool": call.get("name"), "error": repr(e)}
        )
        return ToolMessage(
            f"Error: the tool failed with {type(e).__name__}: {str(e)[:1000]}",
            tool_call_id=call.get("id") or "",
            name=call.get("name"),
            status="error",
        )

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except (GraphBubbleUp, RunStopped):
            raise
        except Exception as e:
            return self._error_result(request, e)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except (GraphBubbleUp, RunStopped):
            raise
        except Exception as e:  # CancelledError is a BaseException and passes through
            return self._error_result(request, e)
