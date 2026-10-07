"""Budgets (B9): limits on tokens, cost, tool calls and time for every agent run.

Each run gets a :class:`Budget`: one chat request (``budgets.turn``) or one ``/plan`` run
(``budgets.plan`` — planning, every task agent and the judge share it). It reaches the agents
through a context variable (:func:`budget_scope`), so nested agents and worker threads see it.

:class:`BudgetMiddleware` (on every agent) enforces it:

- **after each model call** the tokens are counted (the provider's ``usage_metadata``, else an
  estimate) and priced with ``budgets.prices``;
- **before each model call** a reached limit or a stop request ends the run with a short
  "Stopped: …" message (graph jump to the end — the conversation stays valid);
- **before each tool call** the tool is refused with an error result instead of running.
  The tool-call limit is gentler: the model gets one more call to tell the user what it did
  and what is left, then the run stops.

Hard stops — the kill switch (:mod:`kartrix.security.kill_switch`) and the time limit — also
kill a running command (polled every half second) and cancel an in-flight model request.
Time spent waiting for the user (:func:`waiting_for_user`) doesn't count toward ``max_seconds``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from kartrix.config import BudgetLimits, ModelPrice, settings
from kartrix.core.events import ModelCall, emit
from kartrix.observability.logger import get_logger
from kartrix.security import audit
from kartrix.security.kill_switch import STOP_COMMAND_REASON, stop_file, stop_requested_at

logger = get_logger(__name__)

RunKind = Literal["turn", "plan"]


class RunStopped(Exception):
    """The run was stopped by a budget limit or the kill switch."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _fmt_tokens(n: int) -> str:
    return f"{n / 1_000_000:.2f}M" if n >= 1_000_000 else f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Budget:
    """Usage and limits of one run. Thread-safe (sync tools and the planner run in worker threads)."""

    def __init__(
        self,
        kind: RunKind,
        limits: BudgetLimits,
        prices: dict[str, ModelPrice] | None = None,
    ) -> None:
        self.kind = kind
        self.limits = limits
        self.prices = prices or {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.tool_calls = 0
        self.model_calls = 0
        self.estimated = False  # some token counts were estimated (provider sent no usage)
        self.unpriced: set[str] = set()  # models without a price: cost limit not enforced for them
        self.stop_reason: str | None = None
        self.hard = False  # kill switch / time limit: kill commands, cancel model requests
        self.refused_tools = 0
        self._wrap_up_used = False
        self._started = time.monotonic()
        self._started_wall = time.time()
        self._paused = 0.0
        self._waiting = 0
        self._wait_started = 0.0
        self._stop_file = stop_file()
        self._lock = threading.RLock()

    @classmethod
    def for_run(cls, kind: RunKind) -> Budget:
        cfg = settings.budgets
        return cls(kind, cfg.turn if kind == "turn" else cfg.plan, cfg.prices)

    # ── usage ────────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def elapsed(self) -> float:
        """Seconds the run has been working (time waiting for the user excluded)."""
        with self._lock:
            waiting = time.monotonic() - self._wait_started if self._waiting else 0.0
            return time.monotonic() - self._started - self._paused - waiting

    def add_model_call(self, model: str, input_tokens: int, output_tokens: int, estimated: bool = False) -> None:
        price = self.prices.get(model)
        with self._lock:
            self.model_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.estimated = self.estimated or estimated
            if price is None:
                self.unpriced.add(model)
            else:
                self.cost_usd += (input_tokens * price.input + output_tokens * price.output) / 1_000_000

    def add_tool_call(self) -> None:
        with self._lock:
            self.tool_calls += 1

    @property
    def waiting_for_user(self) -> bool:
        return self._waiting > 0

    @contextmanager
    def waiting(self) -> Iterator[None]:
        """Time inside doesn't count toward ``max_seconds`` (an approval prompt, the plan review)."""
        with self._lock:
            if self._waiting == 0:
                self._wait_started = time.monotonic()
            self._waiting += 1
        try:
            yield
        finally:
            with self._lock:
                self._waiting -= 1
                if self._waiting == 0:
                    self._paused += time.monotonic() - self._wait_started

    # ── limits ───────────────────────────────────────────────────────

    def stop(self, reason: str, hard: bool = False) -> None:
        """Stop the run at its next step (the first reason wins)."""
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason
                logger.info("Run stopped", extra={"kind": self.kind, "reason": reason, "usage": self.usage()})
            self.hard = self.hard or hard

    def check(self) -> str | None:
        """The reason the run must stop now (a limit reached or a stop requested), else None."""
        if self.stop_reason is None:
            lim = self.limits
            requested = stop_requested_at(self._stop_file)
            if requested is not None and requested >= self._started_wall:
                self.stop(STOP_COMMAND_REASON, hard=True)
            elif lim.max_seconds is not None and self.elapsed() >= lim.max_seconds:
                self.stop(f"time budget reached ({lim.max_seconds:.0f} s)", hard=True)
            elif lim.max_tokens is not None and self.total_tokens >= lim.max_tokens:
                self.stop(f"token budget reached ({self.total_tokens:,} of {lim.max_tokens:,} tokens)")
            elif lim.max_cost_usd is not None and self.cost_usd >= lim.max_cost_usd:
                self.stop(f"cost budget reached (${self.cost_usd:.2f} of ${lim.max_cost_usd:.2f})")
        return self.stop_reason

    def hard_stop_reason(self) -> str | None:
        """Like :meth:`check`, but only for stops that kill commands and cancel model requests."""
        reason = self.check()
        return reason if self.hard else None

    @property
    def tool_limit_reached(self) -> bool:
        lim = self.limits.max_tool_calls
        return lim is not None and self.tool_calls >= lim

    def check_before_model(self) -> str | None:
        """:meth:`check`, plus: after tools were refused for the tool-call limit, one last model
        call is allowed (to summarise for the user); the next one stops the run."""
        reason = self.check()
        if reason is None and self.tool_limit_reached and self.refused_tools:
            with self._lock:
                if self._wrap_up_used:
                    self.stop(f"tool-call budget reached ({self.limits.max_tool_calls} calls)")
                    reason = self.stop_reason
                else:
                    self._wrap_up_used = True
        return reason

    def exhausted(self) -> str | None:
        """Why the run can't go on: stopped, or out of tool calls with the model already told."""
        reason = self.check()
        if reason is None and self.tool_limit_reached and self.refused_tools:
            return f"tool-call budget reached ({self.limits.max_tool_calls} calls)"
        return reason

    # ── reporting ────────────────────────────────────────────────────

    def usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_estimated": self.estimated,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_models": sorted(self.unpriced),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "seconds": round(self.elapsed(), 1),
        }

    def summary(self) -> str:
        est = "~" if self.estimated else ""
        cost = f"${self.cost_usd:.4f}"
        if self.unpriced:
            cost += f" (cost unknown for {', '.join(sorted(self.unpriced))})"
        return (
            f"{est}{_fmt_tokens(self.total_tokens)} tokens ({est}{_fmt_tokens(self.input_tokens)} in / "
            f"{est}{_fmt_tokens(self.output_tokens)} out) · {self.model_calls} model calls · "
            f"{self.tool_calls} tool calls · {self.elapsed():.0f} s · {cost}"
        )

    async def record_stop(self, session_id: str | None = None) -> None:
        """Audit row for a stopped run (no-op if it wasn't stopped)."""
        if self.stop_reason is None:
            return
        await audit.record(
            actor="system",
            action="budget.stop",
            target=self.kind,
            outcome="stopped",
            session_id=session_id,
            details={"reason": self.stop_reason, "limits": self.limits.model_dump(), **self.usage()},
        )


# ── the current run ───────────────────────────────────────────────────

_current: ContextVar[Budget | None] = ContextVar("kartrix_budget", default=None)


@contextmanager
def budget_scope(budget: Budget) -> Iterator[Budget]:
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def current() -> Budget | None:
    return _current.get()


@contextmanager
def waiting_for_user() -> Iterator[None]:
    """Pause the current run's clock while waiting for the user (no-op outside a run)."""
    budget = _current.get()
    if budget is None:
        yield
        return
    with budget.waiting():
        yield


def raise_if_stopped() -> None:
    """Raise :class:`RunStopped` if the current run can't go on (e.g. before starting another task)."""
    budget = _current.get()
    if budget is not None and (reason := budget.exhausted()):
        raise RunStopped(reason)


def command_stop_check() -> Callable[[], str | None] | None:
    """For long commands: returns a hard-stop reason once the run must be killed."""
    budget = _current.get()
    return budget.hard_stop_reason if budget is not None else None


# ── middleware ────────────────────────────────────────────────────────


def _model_name(model: Any) -> str:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _response_messages(response: Any) -> list[BaseMessage]:
    if isinstance(response, AIMessage):
        return [response]
    inner = getattr(response, "model_response", response)  # ExtendedModelResponse
    return list(getattr(inner, "result", []) or [])


def _account(request: ModelRequest, response: Any, started: float) -> None:
    """Count the call against the run's budget (if any) and emit it for the trace."""
    ai = next((m for m in _response_messages(response) if isinstance(m, AIMessage)), None)
    usage = getattr(ai, "usage_metadata", None) if ai is not None else None
    model = _model_name(request.model)
    if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
        tokens_in, tokens_out, estimated = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), False
    else:
        prompt: list[BaseMessage] = [request.system_message] if request.system_message else []
        prompt += list(request.messages)
        tokens_in, tokens_out, estimated = (
            count_tokens_approximately(prompt), count_tokens_approximately([ai]) if ai is not None else 0, True
        )  # fmt: skip
    if (budget := _current.get()) is not None:
        budget.add_model_call(model, tokens_in, tokens_out, estimated=estimated)
    meta = (ai.response_metadata or {}) if ai is not None else {}
    emit(
        ModelCall(
            model=model, input_tokens=tokens_in, output_tokens=tokens_out, estimated=estimated,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            finish_reason=meta.get("finish_reason") or meta.get("stop_reason"),
        )
    )  # fmt: skip


def _stopped_message(budget: Budget, reason: str) -> AIMessage:
    return AIMessage(content=f"Stopped: {reason}. Used so far: {budget.summary()}.")


def _refusal(request: ToolCallRequest, text: str) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(content=text, name=call.get("name"), tool_call_id=call.get("id") or "", status="error")


class BudgetMiddleware(AgentMiddleware):
    """Count every model call against the current run's budget and stop the run when it is spent
    (nothing to enforce outside a :func:`budget_scope`). Every model call is also emitted as a
    :class:`~kartrix.core.events.ModelCall` event for the run's trace."""

    def _before(self) -> dict[str, Any] | None:
        budget = _current.get()
        if budget is None:
            return None
        reason = budget.check_before_model()
        if reason is None:
            return None
        return {"jump_to": "end", "messages": [_stopped_message(budget, reason)]}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._before()

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._before()

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        started = time.perf_counter()
        response = handler(request)
        _account(request, response, started)
        return response

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]) -> Any:
        started = time.perf_counter()
        response = await handler(request)
        _account(request, response, started)
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        budget = _current.get()
        if budget is None:
            return await handler(request)
        if reason := budget.check():
            audit.note(budget=reason)
            return _refusal(request, f"Error: stopped — {reason}. The tool was not run.")
        if budget.tool_limit_reached:
            with budget._lock:
                budget.refused_tools += 1
            audit.note(budget="tool-call budget reached")
            return _refusal(
                request,
                f"Error: stopped — the tool-call budget of this run ({budget.limits.max_tool_calls} calls) is used "
                "up; the tool was not run. Don't call more tools: tell the user what you did and what is left.",
            )
        budget.add_tool_call()
        return await handler(request)
