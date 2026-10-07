"""Kill switch (B9): stop a running agent now, from this terminal or any other.

- **Ctrl+C during a run** (:func:`run_stoppable`): the first press stops the run — the budget is
  marked stopped (no further model or tool call starts, the running command's process tree is
  killed within half a second) and the run's task is cancelled, so an in-flight model request
  is abandoned too. Kartrix stays open. A second press quits Kartrix. While the run waits for
  the user's answer (approval prompt) it is not cancelled: the prompt is answered "no" and the
  run ends at its next step.
- **``kartrix stop``** from any terminal (:func:`request_stop`): writes the current time to a stop
  file in the per-user data folder. Every run that started before that moment stops (checked
  before each model/tool call and every half second while a command runs); runs started later
  are unaffected, so the file never needs clearing.

This module imports nothing heavy: ``kartrix stop`` must work instantly, even with a broken config.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kartrix.paths import user_data_dir

if TYPE_CHECKING:
    from kartrix.security.budget import Budget


CTRL_C_REASON = "stopped by the user (Ctrl+C)"
STOP_COMMAND_REASON = "stopped with `kartrix stop`"
POLL_SECONDS = 0.5
_STOP_FILE = "stop"


def stop_file() -> Path:
    return user_data_dir() / _STOP_FILE


def request_stop() -> float:
    """Ask every Kartrix run in progress on this machine (for this user) to stop. Returns the time."""
    now = time.time()
    path = stop_file()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(repr(now), encoding="utf-8")
    tmp.replace(path)
    return now


def stop_requested_at(path: Path) -> float | None:
    """When ``kartrix stop`` was last run (None if never)."""
    try:
        return float(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def close_dangling_tool_calls(agent: Any, config: dict[str, Any], reason: str) -> int:
    """After a run was cancelled, answer the tool calls of its last model turn that never got a
    result (providers reject a conversation with unanswered tool calls). Returns how many."""
    from langchain_core.messages import AIMessage, ToolMessage

    state = await agent.aget_state(config)
    if getattr(state, "interrupts", None):
        return 0  # paused at an approval: it is asked again later, nothing to repair
    messages = (state.values or {}).get("messages", [])
    last_ai = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)), None)
    if last_ai is None or not messages[last_ai].tool_calls:
        return 0
    answered = {m.tool_call_id for m in messages[last_ai + 1 :] if isinstance(m, ToolMessage)}
    missing = [
        ToolMessage(
            content=f"Error: stopped — {reason}. The tool was not run (or its result was discarded).",
            name=call["name"],
            tool_call_id=call["id"] or "",
            status="error",
        )
        for call in messages[last_ai].tool_calls
        if call.get("id") and call["id"] not in answered
    ]
    if missing:
        await agent.aupdate_state(config, {"messages": [*missing, AIMessage(content=f"Stopped: {reason}.")]})
    return len(missing)


async def run_stoppable[T](
    coro: Coroutine[Any, Any, T], budget: Budget, on_stop: Callable[[str], None] | None = None
) -> T | None:
    """Run ``coro`` inside ``budget`` as its own task; None if the kill switch cancelled it.

    ``on_stop(message)`` is called (on the event loop) when a stop is requested, e.g. to print it."""
    from kartrix.security.budget import budget_scope

    loop = asyncio.get_running_loop()
    with budget_scope(budget):
        task: asyncio.Task[T] = asyncio.ensure_future(coro)  # copies the context: the budget is visible inside
    cancelled = False

    def cancel_once() -> None:
        nonlocal cancelled
        if not cancelled and not task.done():
            cancelled = True
            task.cancel()

    def stop_now(reason: str) -> None:
        budget.stop(reason, hard=True)
        if budget.waiting_for_user:
            if on_stop:
                on_stop("Stopping — answer or press Enter at the prompt; it counts as 'no'.")
            return
        if on_stop:
            on_stop("Stopping…")
        cancel_once()

    presses = 0
    previous: Any = None

    def on_sigint(signum: int, frame: Any) -> None:
        nonlocal presses
        presses += 1
        if presses > 1:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        loop.call_soon_threadsafe(stop_now, CTRL_C_REASON)

    async def watch() -> None:
        # `kartrix stop` and the time limit stop a run even while a model request is in flight.
        while not task.done():
            await asyncio.sleep(POLL_SECONDS)
            reason = budget.hard_stop_reason()
            if reason and not budget.waiting_for_user and not cancelled:
                if on_stop:
                    on_stop(f"Stopping: {reason}…")
                cancel_once()

    main_thread = threading.current_thread() is threading.main_thread()
    if main_thread:
        previous = signal.signal(signal.SIGINT, on_sigint)
    watcher = asyncio.ensure_future(watch())
    try:
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if cancelled and not (current and current.cancelling()):
            return None  # we cancelled it; the caller's own task wasn't cancelled
        raise
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        if main_thread and presses <= 1:
            signal.signal(signal.SIGINT, previous)
