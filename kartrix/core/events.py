"""The event stream (C1): everything a run does, as typed events instead of terminal output.

The core never prints or reads from the terminal. It :func:`emit`\\ s events; front ends subscribe:
the REPL renders them with Rich (``kartrix.ui.console``), headless mode writes them as JSON lines
and builds its report from them (``kartrix.headless``). Questions for the user go through the
:mod:`kartrix.core.interaction` callbacks, not through events.

Events are emitted from the event loop and from worker threads (sync tools), so subscribers
must be quick and thread-safe. A failing subscriber is logged and never breaks the run.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts: float = Field(default_factory=time.time)
    run_id: str | None = None  # set by emit() from the enclosing run_scope()


class Notice(_Event):
    """A status line: startup info, warnings, hints."""

    type: Literal["notice"] = "notice"
    level: Literal["info", "success", "warning", "error"] = "info"
    text: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False  # some token counts were estimated (provider sent no usage)
    cost_usd: float = 0.0
    cost_known: bool = True  # False: a model had no price, so the cost is a lower bound
    model_calls: int = 0
    tool_calls: int = 0
    seconds: float = 0.0


class RunStarted(_Event):
    type: Literal["run_started"] = "run_started"
    kind: Literal["ask", "plan"]
    input: str
    session_id: str | None = None


RunStatus = Literal["completed", "failed", "stopped", "error"]


class RunFinished(_Event):
    type: Literal["run_finished"] = "run_finished"
    kind: Literal["ask", "plan"]
    status: RunStatus
    detail: str | None = None  # why it stopped / failed
    usage: Usage | None = None


class AssistantMessage(_Event):
    """The agent's answer to a chat request."""

    type: Literal["assistant_message"] = "assistant_message"
    text: str
    cached: bool = False  # served from the semantic cache, the agent didn't run


class AssistantDelta(_Event):
    """A piece of the answer while the model is still writing it (token streaming). The complete text
    follows as :class:`AssistantMessage`; JSONL event logs leave the deltas out."""

    type: Literal["assistant_delta"] = "assistant_delta"
    text: str


class ModelCall(_Event):
    """One model call finished (for traces): which model, tokens, latency, why it stopped."""

    type: Literal["model_call"] = "model_call"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False  # tokens counted by Kartrix because the provider sent no usage
    duration_ms: float = 0.0
    finish_reason: str | None = None


class ToolCallStarted(_Event):
    type: Literal["tool_call_started"] = "tool_call_started"
    call_id: str | None
    tool: str
    target: str | None = None
    task_key: str | None = None
    args: dict[str, Any] | None = None  # secrets redacted, long strings clipped


class ToolCallFinished(_Event):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    call_id: str | None
    tool: str
    outcome: str  # ok, error, invalid_args, denied, declined, stopped, …
    duration_ms: float
    task_key: str | None = None
    output_chars: int | None = None


class AgentStep(_Event):
    """A subagent of the chat graph started or finished (router, explorer, coder, reviewer, responder)."""

    type: Literal["agent_step"] = "agent_step"
    agent: str
    status: Literal["started", "finished"]
    summary: str | None = None


class ContextSection(BaseModel):
    name: str
    tokens: int
    budget: int
    items: int = 0  # memories / turns included
    dropped: int = 0  # items left out (or 1 if the text was cut)


class ContextAssembled(_Event):
    """The context for this turn was assembled: what went in, and what the budgets cut."""

    type: Literal["context_assembled"] = "context_assembled"
    sections: list[ContextSection]
    stale: int = 0  # recalled memories that mention files changed since

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sections)


class ApprovalRequested(_Event):
    type: Literal["approval_requested"] = "approval_requested"
    tool_call_id: str
    tool: str
    command: str
    directory: str
    category: str
    reason: str
    task_key: str | None = None


class ApprovalResolved(_Event):
    type: Literal["approval_resolved"] = "approval_resolved"
    tool_call_id: str
    decision: str  # approve, approve_session, edit, reject
    by: str  # user, policy
    message: str | None = None


class PlanProposed(_Event):
    type: Literal["plan_proposed"] = "plan_proposed"
    plan: dict[str, Any]


class PlanReviewed(_Event):
    type: Literal["plan_reviewed"] = "plan_reviewed"
    approved: bool
    by: str  # user, policy
    feedback: str | None = None


class ProjectStarted(_Event):
    type: Literal["project_started"] = "project_started"
    project_id: str
    resumed: bool
    recovered: int = 0  # crashed tasks reset on resume


class TaskStarted(_Event):
    type: Literal["task_started"] = "task_started"
    task_key: str
    title: str


class TaskFinished(_Event):
    type: Literal["task_finished"] = "task_finished"
    task_key: str
    title: str
    status: Literal["completed", "failed", "stopped", "skipped"]
    detail: str | None = None


class Progress(_Event):
    type: Literal["progress"] = "progress"
    completed: int
    total: int
    in_progress: int = 0
    pending: int = 0
    failed: int = 0
    blocked: int = 0


class ProjectFinished(_Event):
    type: Literal["project_finished"] = "project_finished"
    project_id: str
    status: Literal["completed", "failed", "stopped", "planned"]
    counts: dict[str, int] = Field(default_factory=dict)


class FilesChanged(_Event):
    """A checkpointed step changed files (undoable with /undo)."""

    type: Literal["files_changed"] = "files_changed"
    label: str
    files: list[str]
    count: int


Event = Annotated[
    Notice | RunStarted | RunFinished | AssistantMessage | AssistantDelta | ModelCall | ToolCallStarted | ToolCallFinished | AgentStep
    | ContextAssembled | ApprovalRequested
    | ApprovalResolved | PlanProposed | PlanReviewed | ProjectStarted | TaskStarted | TaskFinished | Progress
    | ProjectFinished | FilesChanged,
    Field(discriminator="type"),
]  # fmt: skip
EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Add ``fn``; returns a function that removes it again."""
        with self._lock:
            self._subscribers.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return unsubscribe

    def emit(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception as e:
                logger.error("Event subscriber failed", extra={"event": event.type, "error": repr(e)})


bus = EventBus()
_run_id: ContextVar[str | None] = ContextVar("kartrix_run_id", default=None)


def emit(event: Event) -> None:
    """Send ``event`` to every subscriber, tagged with the current run id."""
    run_id = _run_id.get()
    if run_id is not None and event.run_id is None:
        event = event.model_copy(update={"run_id": run_id})
    bus.emit(event)


def notice(text: str, level: Literal["info", "success", "warning", "error"] = "info") -> None:
    emit(Notice(text=text, level=level))


@contextmanager
def run_scope(run_id: str) -> Iterator[None]:
    """Tag every event emitted inside (including worker threads started inside) with ``run_id``."""
    token = _run_id.set(run_id)
    try:
        yield
    finally:
        _run_id.reset(token)


@contextmanager
def collecting() -> Iterator[list[Event]]:
    """Collect the events emitted inside (tests, reports)."""
    seen: list[Event] = []
    unsubscribe = bus.subscribe(seen.append)
    try:
        yield seen
    finally:
        unsubscribe()
