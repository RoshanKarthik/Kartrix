"""Token streaming of the answer.

Only the responder's model call streams: :class:`AnswerStream` is attached to that one call, so the
explorer, coder and reviewer (whose replies carry tool calls) keep the plain request/response path,
and nothing about tool-call parsing changes. Each text token becomes an :class:`AssistantDelta`
event; the REPL prints them as they arrive and the complete :class:`AssistantMessage` follows.
Tool-call markup the model writes into its answer (``kartrix.agent.tool_markup``) is never streamed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeVar
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from kartrix.agent.tool_markup import MarkupFilter
from kartrix.core.events import AssistantDelta, emit

T = TypeVar("T")


class AnswerStream(AsyncCallbackHandler):
    """Emits every text token of the model calls it is attached to. Having ``tap_output_*`` makes it a
    streaming handler for LangChain, which then calls the provider's streaming API."""

    def __init__(self) -> None:
        self._calls: dict[UUID, MarkupFilter] = {}  # per model call: a retry or a nudge streams afresh

    def tap_output_aiter(self, run_id: UUID, output: AsyncIterator[T]) -> AsyncIterator[T]:
        return output

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        return output

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: Any = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(token, list):  # content blocks: only the text ones are the answer
            token = "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in token
                            if isinstance(b, str) or b.get("type") == "text")  # fmt: skip
        if token:  # reasoning models send their thinking as empty-text chunks
            self._emit(self._calls.setdefault(run_id, MarkupFilter()).feed(token))

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if call := self._calls.pop(run_id, None):
            self._emit(call.end())

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._calls.pop(run_id, None)  # what was held back is dropped with the failed call

    @staticmethod
    def _emit(text: str) -> None:
        if text:
            emit(AssistantDelta(text=text))
