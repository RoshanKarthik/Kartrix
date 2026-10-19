"""``kartrix worker``: one workspace's agent, driven by the server over stdin/stdout (JSON lines,
:mod:`kartrix.server.protocol`).

The server starts it with the workspace as its working directory and ``KARTRIX_USER_ID`` set, so
the core's per-process state (workspace jail, sandbox, index, event bus) belongs to exactly one
workspace of one user. Approvals and plan reviews are sent to the server and wait for its answer.
Anything the core prints goes to stderr; stdout carries only protocol messages. The protocol pipes are
private to this process: child processes (git for checkpoints, sandboxed commands, MCP servers) get
``NUL`` as stdin — on Windows a child inheriting a pipe this process is blocking on hangs at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import uuid
from typing import Any, BinaryIO, TextIO

from pydantic import ValidationError

from kartrix.server.protocol import OP_ADAPTER, ApprovalOp, PlanReviewOp, RunOp, ShutdownOp, StopOp, decode, encode

STOP_REASON = "stopped from the API"


class Channel:
    """Writes protocol messages to the server; safe from worker threads (events of sync tools)."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:
        line = encode(message)
        with self._lock:
            try:
                self._out.write(line)
                self._out.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass  # the server went away; the stdin reader ends the worker


class RemoteFrontEnd:
    """The worker's approver and plan reviewer: ask the server, wait for the answer it forwards."""

    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self._pending: dict[str, tuple[str, asyncio.Future[Any]]] = {}

    async def approve(self, requests: list[Any]) -> list[Any]:
        from dataclasses import asdict

        from kartrix.security.approvals import ApprovalDecision
        from kartrix.security.budget import current

        if (run := current()) is not None and run.stop_reason:
            return [
                ApprovalDecision("reject", message=f"no answer — run stopped ({run.stop_reason})") for _ in requests
            ]
        answer = await self._ask("approval", {"requests": [asdict(r) for r in requests]})
        decisions = answer.decisions if isinstance(answer, ApprovalOp) else []
        if len(decisions) != len(requests):
            return [ApprovalDecision("reject", message="no answer — run stopped") for _ in requests]
        return [ApprovalDecision(d.type, command=d.command, message=d.message) for d in decisions]

    async def review_plan(self, plan: Any) -> Any:
        from kartrix.core.interaction import PlanReview
        from kartrix.security.budget import RunStopped
        from kartrix.tasks.planner import ExecutionPlan

        answer = await self._ask("plan_review", {"plan": plan.model_dump(mode="json")})
        if not isinstance(answer, PlanReviewOp):
            raise RunStopped(STOP_REASON)
        if not answer.approved:
            return PlanReview(None, answer.feedback)
        if answer.plan is None:
            return PlanReview(plan)
        try:
            return PlanReview(ExecutionPlan.model_validate(answer.plan))
        except ValidationError:
            return PlanReview(plan)  # an edit that doesn't parse: approve what was proposed

    async def _ask(self, what: str, payload: dict[str, Any]) -> Any:
        request_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (what, future)
        self.channel.send({"kind": f"{what}_needed", "request_id": request_id, **payload})
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    def answer(self, op: ApprovalOp | PlanReviewOp) -> bool:
        what = "approval" if isinstance(op, ApprovalOp) else "plan_review"
        pending = self._pending.get(op.request_id)
        if pending is None or pending[0] != what or pending[1].done():
            return False
        pending[1].set_result(op)
        return True

    def cancel_all(self) -> None:
        """Stop: every open question is answered with "no" (approvals) or ends planning."""
        for _, future in list(self._pending.values()):
            if not future.done():
                future.set_result(None)


def _read_protocol(source: BinaryIO, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[bytes | None]) -> None:
    for line in source:
        loop.call_soon_threadsafe(queue.put_nowait, line)
    loop.call_soon_threadsafe(queue.put_nowait, None)  # the server closed our stdin: shut down


async def _serve(channel: Channel, source: BinaryIO, session_id: str, mcp: bool, model: str | None = None) -> int:
    from kartrix.core.events import bus
    from kartrix.core.session import CoreSession, StartOptions

    unsubscribe = bus.subscribe(lambda event: channel.send({"kind": "event", "event": event.model_dump(mode="json")}))
    front = RemoteFrontEnd(channel)
    options = StartOptions(session_id=session_id, mcp=mcp, resume_pending=False, model=model)
    try:
        core = await CoreSession.start(front.approve, front.review_plan, options)
    except Exception as e:
        channel.send({"kind": "start_failed", "detail": f"{type(e).__name__}: {e}"})
        unsubscribe()
        return 1
    channel.send({"kind": "ready"})

    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue[bytes | None] = asyncio.Queue()
    reader = threading.Thread(target=_read_protocol, args=(source, loop, inbox), name="kartrix-worker-in", daemon=True)
    reader.start()
    running: asyncio.Task[None] | None = None

    async def do_run(op: RunOp) -> None:
        try:
            if op.session_id != core.session_id:
                await core.use_session(op.session_id)
            if op.model is not None:
                core.set_model(op.model)  # rebuilds the agent only when it changes
            if op.kind == "ask":
                outcome = await core.ask(op.input)
            else:
                outcome = await core.plan(op.input)
            message = {
                "status": outcome.status,
                "detail": outcome.detail,
                "answer": outcome.answer,
                "usage": outcome.usage.model_dump(mode="json"),
                "files_changed": outcome.files_changed,
            }
        except Exception as e:  # the core reports its own errors; this is a bug or a dead database
            message = {"status": "error", "detail": f"{type(e).__name__}: {e}"}
        channel.send({"kind": "run_finished", "run_id": op.run_id, **message})

    try:
        while True:
            line = await inbox.get()
            if line is None:
                break
            try:
                op = OP_ADAPTER.validate_python(decode(line))
            except (ValueError, ValidationError) as e:
                channel.send({"kind": "event", "event": {"type": "notice", "level": "error", "text": f"bad op: {e}"}})
                continue
            if isinstance(op, ShutdownOp):
                break
            if isinstance(op, RunOp):
                if running is not None and not running.done():
                    channel.send({"kind": "run_finished", "run_id": op.run_id, "status": "error", "detail": "busy"})
                    continue
                running = asyncio.create_task(do_run(op), name=f"kartrix-run-{op.run_id}")
            elif isinstance(op, ApprovalOp | PlanReviewOp):
                front.answer(op)
            elif isinstance(op, StopOp) and running is not None and not running.done():
                if core.last_budget is not None:
                    core.last_budget.stop(STOP_REASON, hard=True)
                front.cancel_all()
    finally:
        if running is not None and not running.done():
            if core.last_budget is not None:
                core.last_budget.stop(STOP_REASON, hard=True)
            front.cancel_all()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=30)
        unsubscribe()
        await core.close()
    return 0


def run_worker(session_id: str, mcp: bool = False, model: str | None = None) -> int:
    """Entry point of ``kartrix worker`` (started by the server, never by hand)."""
    protocol_in, protocol_out = private_protocol_pipes()
    return asyncio.run(_serve(Channel(protocol_out), protocol_in, session_id, mcp, model))


def private_protocol_pipes() -> tuple[BinaryIO, TextIO]:
    """Take stdin/stdout for the protocol, out of reach of child processes.

    ``os.dup()`` handles are not inherited; fds 0 and 1 (which children do get) are pointed elsewhere:
    stdin at NUL, stdout at stderr (stray prints, also from C code)."""
    protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb")
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, sys.stdin.fileno())
    os.close(null)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdin = open(os.devnull, encoding="utf-8")
    sys.stdout = sys.stderr
    return protocol_in, protocol_out
