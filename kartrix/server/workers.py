"""Worker processes and run feeds, server side.

- :class:`WorkerProcess` — one ``kartrix worker`` subprocess per active workspace (its own process group,
  the workspace as working directory, ``KARTRIX_USER_ID`` set). Its stdout is read by a thread and handed to
  the event loop line by line; its stderr goes to a log file.
- :class:`RunFeed` — the messages of one run, in order. Every message but ``assistant_delta`` events gets a
  sequence number and is written to ``run_events``; live subscribers (SSE streams) get a backlog from memory
  plus a queue, set up without an ``await`` in between, so nothing is lost or sent twice.
- :class:`WorkerPool` — starts runs, routes answers, stops and workspace commands (``/undo``, ``/mode`` …),
  shuts idle workers down.

Threads touch nothing here except through ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from sqlalchemy import update

from kartrix.config import settings
from kartrix.db.engine import session_scope
from kartrix.db.models import Run, RunEvent
from kartrix.observability.logger import get_logger
from kartrix.server.protocol import (
    ApprovalOp,
    CommandOp,
    Decision,
    PlanReviewOp,
    RunOp,
    ShutdownOp,
    StopOp,
    decode,
    encode,
)

logger = get_logger(__name__)

COMMAND_SECONDS = 60.0
CONNECT_SECONDS = 300.0  # /connect may download the MCP server first


class PoolError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class FeedItem:
    seq: int | None  # None: live only (streamed answer tokens), never stored or replayed
    kind: str
    data: dict[str, Any]


@dataclass
class RunFeed:
    run_id: uuid.UUID
    workspace_id: uuid.UUID
    items: list[FeedItem] = field(default_factory=list)
    subscribers: set[asyncio.Queue[FeedItem | None]] = field(default_factory=set)
    pending: dict[str, str] = field(default_factory=dict)  # open question id → "approval" | "plan_review"
    finished: bool = False  # run_finished published, subscribers released
    ending: bool = False  # the run ended; its row is being written before run_finished goes out
    stop_requested: bool = False
    _seq: int = 0

    def publish(self, kind: str, data: dict[str, Any], persist: bool = True) -> FeedItem:
        if persist:
            self._seq += 1
        item = FeedItem(self._seq if persist else None, kind, data)
        if persist:
            self.items.append(item)
        for queue in self.subscribers:
            queue.put_nowait(item)
        return item

    def subscribe(self, after: int) -> tuple[list[FeedItem], asyncio.Queue[FeedItem | None]]:
        """The stored items after ``after`` and a queue of what comes next (None: the run ended)."""
        queue: asyncio.Queue[FeedItem | None] = asyncio.Queue()
        if self.finished:
            queue.put_nowait(None)
        else:
            self.subscribers.add(queue)
        return [i for i in self.items if i.seq is not None and i.seq > after], queue

    def unsubscribe(self, queue: asyncio.Queue[FeedItem | None]) -> None:
        self.subscribers.discard(queue)

    def close(self) -> None:
        self.finished = True
        for queue in self.subscribers:
            queue.put_nowait(None)
        self.subscribers.clear()


class WorkerProcess:
    def __init__(self, pool: WorkerPool, workspace_id: uuid.UUID, user_id: uuid.UUID, root: Path) -> None:
        self.pool = pool
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.root = root
        self.proc: subprocess.Popen[bytes] | None = None
        self.ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.feed: RunFeed | None = None
        self.commands: dict[str, asyncio.Future[dict[str, Any]]] = {}  # open command id → its answer
        self.last_active = time.monotonic()
        self.exited = False
        self._stdin_lock = threading.Lock()
        self._log: IO[bytes] | None = None

    @property
    def busy(self) -> bool:
        return self.feed is not None and not self.feed.finished

    def start(self, session_id: str, model: str | None = None, mode: str | None = None) -> None:
        loop = asyncio.get_running_loop()
        logs = self.pool.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._log = open(logs / f"worker-{self.workspace_id}.log", "ab")
        env = {**os.environ, "KARTRIX_USER_ID": str(self.user_id), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        argv = [sys.executable, "-m", "kartrix.cli", "worker", "--session", session_id]
        if model is not None:
            argv += ["--model", model]  # built with it, rather than switched before the first run
        if mode is not None:
            argv += ["--mode", mode]  # what /mode chose for this workspace, before an idle restart
        if settings.server.allow_local_paths:
            argv.append("--mcp")  # the user's own machine: their connected MCP servers too
        group: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )  # Ctrl+C in the server's terminal doesn't reach the workers; the server shuts them down
        self.proc = subprocess.Popen(  # noqa: S603 — our own interpreter and module, fixed arguments
            argv, cwd=self.root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, **group
        )
        logger.info("Worker started", extra={"workspace": str(self.workspace_id), "pid": self.proc.pid})
        threading.Thread(target=self._read, args=(loop,), name=f"kartrix-worker-{self.proc.pid}", daemon=True).start()

    def _read(self, loop: asyncio.AbstractEventLoop) -> None:
        assert self.proc is not None and self.proc.stdout is not None  # noqa: S101
        for line in self.proc.stdout:
            loop.call_soon_threadsafe(self._on_line, line)
        code = self.proc.wait()
        loop.call_soon_threadsafe(self._on_exit, code)

    def send(self, op: Any) -> None:
        if self.proc is None or self.proc.stdin is None or self.exited:
            raise PoolError(409, "the workspace's worker is not running")
        with self._stdin_lock:
            try:
                self.proc.stdin.write(encode(op).encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise PoolError(409, "the workspace's worker is not running") from e

    # ── messages from the worker (event loop) ──────────────────────────

    def _on_line(self, line: bytes) -> None:
        try:
            message = decode(line)
        except ValueError:
            logger.warning("Worker sent a line that isn't JSON", extra={"line": line[:200]})
            return
        kind = message.pop("kind", None)
        if kind == "ready":
            if not self.ready.done():
                self.ready.set_result(None)
            return
        if kind == "start_failed":
            if not self.ready.done():
                self.ready.set_exception(PoolError(500, f"the agent could not start: {message.get('detail')}"))
            return
        if kind == "command_result":
            answer = self.commands.pop(str(message.get("request_id")), None)
            if answer is not None and not answer.done():
                answer.set_result(message)
            return
        feed = self.feed
        if feed is None or feed.finished or feed.ending:
            return  # between runs (e.g. "index ready"): nobody is listening
        if kind == "event":
            event = message.get("event") or {}
            self.pool.store(feed, feed.publish("event", event, persist=event.get("type") != "assistant_delta"))
        elif kind in ("approval_needed", "plan_review_needed"):
            request_id = str(message.get("request_id"))
            feed.pending[request_id] = kind.removesuffix("_needed")
            self.pool.store(feed, feed.publish(kind, message))
        elif kind == "run_finished":
            self.pool.finish(self, feed, message)

    def _on_exit(self, code: int) -> None:
        self.exited = True
        if self._log is not None:
            self._log.close()
        logger.info("Worker exited", extra={"workspace": str(self.workspace_id), "code": code})
        if not self.ready.done():
            self.ready.set_exception(PoolError(500, f"the agent's worker exited while starting (code {code})"))
        for answer in self.commands.values():
            if not answer.done():
                answer.set_exception(PoolError(500, f"the agent's worker exited (code {code})"))
        self.commands.clear()
        if self.feed is not None and not self.feed.finished:
            detail = f"the agent's worker exited (code {code}) — see the server's worker log"
            self.pool.finish(self, self.feed, {"status": "error", "detail": detail})
        self.pool.forget(self)

    # ── lifecycle ──────────────────────────────────────────────────────

    async def shutdown(self, grace: float = 30.0) -> None:
        if self.proc is None or self.exited:
            return
        with contextlib.suppress(PoolError):
            self.send(ShutdownOp())
        try:
            await asyncio.wait_for(asyncio.to_thread(self.proc.wait), grace)
        except TimeoutError:
            self.proc.kill()


class WorkerPool:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.workers: dict[uuid.UUID, WorkerProcess] = {}
        self.feeds: dict[uuid.UUID, RunFeed] = {}
        self.modes: dict[uuid.UUID, str] = {}  # workspace → permission mode set with /mode (until the server restarts)
        self._writes: asyncio.Queue[Callable[[], Awaitable[None]]] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._reaper: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def open(self) -> None:
        self._writes = asyncio.Queue()
        self._writer = asyncio.create_task(self._write_loop(), name="kartrix-run-writer")
        self._reaper = asyncio.create_task(self._reap_loop(), name="kartrix-worker-reaper")

    async def close(self) -> None:
        for worker in list(self.workers.values()):
            await worker.shutdown()
        await asyncio.sleep(0)  # let exit callbacks queue their last writes
        if self._writes is not None:
            await self._writes.join()
        for task in (self._reaper, self._writer):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # ── database writes, in order ─────────────────────────────────────

    def _enqueue(self, write: Callable[[], Awaitable[None]]) -> None:
        assert self._writes is not None, "WorkerPool.open() was not called"  # noqa: S101
        self._writes.put_nowait(write)

    async def _write_loop(self) -> None:
        assert self._writes is not None  # noqa: S101
        while True:
            write = await self._writes.get()
            try:
                await write()
            except Exception as e:
                logger.error("Run feed write failed", extra={"error": repr(e)})
            finally:
                self._writes.task_done()

    def store(self, feed: RunFeed, item: FeedItem) -> None:
        if item.seq is None:
            return
        run_id, seq, kind, data = feed.run_id, item.seq, item.kind, item.data

        async def write() -> None:
            async with session_scope() as s:
                s.add(RunEvent(run_id=run_id, seq=seq, kind=kind, data=data))

        self._enqueue(write)

    # ── runs ───────────────────────────────────────────────────────────

    async def start_run(
        self, *, user_id: uuid.UUID, workspace_id: uuid.UUID, root: Path, run_id: uuid.UUID, op: RunOp
    ) -> RunFeed:
        worker = self.workers.get(workspace_id)
        if worker is not None and worker.busy:
            raise PoolError(409, "a run is already in progress in this workspace")
        if worker is not None and worker.commands:
            raise PoolError(409, "a command is still running in this workspace — try again in a moment")
        worker = await self._worker_for(user_id, workspace_id, root, op.session_id, op.model)
        feed = RunFeed(run_id, workspace_id)
        self.feeds[run_id] = feed
        worker.feed = feed
        worker.last_active = time.monotonic()
        task = asyncio.create_task(self._dispatch(worker, feed, op), name=f"kartrix-dispatch-{run_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return feed

    async def _worker_for(
        self, user_id: uuid.UUID, workspace_id: uuid.UUID, root: Path, session_id: str, model: str | None = None
    ) -> WorkerProcess:
        """The workspace's worker, started if it isn't running."""
        worker = self.workers.get(workspace_id)
        if worker is not None:
            return worker
        if len(self.workers) >= settings.server.max_workers and not await self._evict_one():
            raise PoolError(503, "the server is at its limit of active workspaces — try again shortly")
        worker = WorkerProcess(self, workspace_id, user_id, root)
        self.workers[workspace_id] = worker
        worker.start(session_id, model, self.modes.get(workspace_id))
        return worker

    async def command(
        self,
        *,
        user_id: uuid.UUID,
        workspace_id: uuid.UUID,
        root: Path,
        session_id: str,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a workspace command in the workspace's worker (started if needed) and return its result.
        Commands that change something are refused while a run is in progress."""
        from kartrix.core.commands import COMMANDS

        if name not in COMMANDS:
            raise PoolError(422, f"unknown command {name!r}")
        worker = self.workers.get(workspace_id)
        if worker is not None and worker.busy and COMMANDS[name]:
            raise PoolError(409, "a run is in progress in this workspace — try again when it ends")
        worker = await self._worker_for(user_id, workspace_id, root, session_id)
        try:
            await asyncio.wait_for(asyncio.shield(worker.ready), settings.server.worker_start_timeout)
        except TimeoutError:
            raise PoolError(504, "the agent took too long to start") from None
        request_id = uuid.uuid4().hex
        answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        worker.commands[request_id] = answer
        worker.last_active = time.monotonic()
        try:
            worker.send(CommandOp(request_id=request_id, name=name, args=args))
            timeout = CONNECT_SECONDS if name == "mcp.connect" else COMMAND_SECONDS
            message = await asyncio.wait_for(answer, timeout)
        except TimeoutError:
            raise PoolError(504, f"{name} took too long") from None
        finally:
            worker.commands.pop(request_id, None)
            worker.last_active = time.monotonic()
        if not message.get("ok"):
            status = 409 if message.get("busy") else 422
            raise PoolError(status, str(message.get("error") or f"{name} failed"))
        result: dict[str, Any] = message.get("result") or {}
        if name == "mode" and result.get("changed"):
            self.modes[workspace_id] = str(result["mode"])
        return result

    async def _dispatch(self, worker: WorkerProcess, feed: RunFeed, op: RunOp) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(worker.ready), settings.server.worker_start_timeout)
        except TimeoutError:
            self.finish(worker, feed, {"status": "error", "detail": "the agent took too long to start"})
            await worker.shutdown(grace=5)
            return
        except PoolError as e:
            self.finish(worker, feed, {"status": "error", "detail": e.detail})
            return
        if feed.finished:
            return
        if feed.stop_requested:
            self.finish(worker, feed, {"status": "stopped", "detail": "stopped before it started"})
            return
        try:
            worker.send(op)
        except PoolError as e:
            self.finish(worker, feed, {"status": "error", "detail": e.detail})

    def finish(self, worker: WorkerProcess, feed: RunFeed, message: dict[str, Any]) -> None:
        """The run ended: its row first, then the last feed message — a client that sees ``run_finished``
        and reads the run gets the final status — then the feed is dropped (the DB has it all)."""
        if feed.finished or feed.ending:
            return
        feed.ending = True
        status = message.get("status") if message.get("status") in ("completed", "failed", "stopped") else "error"
        result = {
            "status": status,
            "detail": message.get("detail"),
            "answer": message.get("answer"),
            "usage": message.get("usage"),
            "files_changed": message.get("files_changed") or [],
        }
        feed.pending.clear()
        if worker.feed is feed:
            worker.feed = None
        worker.last_active = time.monotonic()
        run_id = feed.run_id

        async def write() -> None:
            async with session_scope() as s:
                await s.execute(
                    update(Run)
                    .where(Run.id == run_id)
                    .values(
                        status=status,
                        detail=result["detail"],
                        answer=result["answer"],
                        usage=result["usage"],
                        finished_at=datetime.now(UTC),
                    )
                )
            item = feed.publish("run_finished", result)
            feed.close()
            async with session_scope() as s:
                s.add(RunEvent(run_id=run_id, seq=item.seq, kind=item.kind, data=item.data))
            self.feeds.pop(run_id, None)  # every event of the run is in the database now

        self._enqueue(write)

    def answer(self, run_id: uuid.UUID, request_id: str, op: ApprovalOp | PlanReviewOp) -> None:
        feed = self.feeds.get(run_id)
        expected = "approval" if isinstance(op, ApprovalOp) else "plan_review"
        if feed is None or feed.finished or feed.ending or feed.pending.get(request_id) != expected:
            raise PoolError(404, "no such open question in this run")
        worker = self.workers.get(feed.workspace_id)
        if worker is None:
            raise PoolError(409, "the workspace's worker is not running")
        worker.send(op)
        del feed.pending[request_id]

    def approve(self, run_id: uuid.UUID, request_id: str, decisions: list[Decision]) -> None:
        self.answer(run_id, request_id, ApprovalOp(request_id=request_id, decisions=decisions))

    def stop(self, run_id: uuid.UUID) -> None:
        feed = self.feeds.get(run_id)
        if feed is None or feed.finished or feed.ending:
            raise PoolError(409, "the run is not in progress")
        feed.stop_requested = True
        feed.pending.clear()
        worker = self.workers.get(feed.workspace_id)
        if worker is not None and worker.ready.done() and not worker.exited:
            worker.send(StopOp())

    def forget(self, worker: WorkerProcess) -> None:
        if self.workers.get(worker.workspace_id) is worker:
            del self.workers[worker.workspace_id]

    # ── idle workers ──────────────────────────────────────────────────

    async def _evict_one(self) -> bool:
        idle = [w for w in self.workers.values() if not w.busy]
        if not idle:
            return False
        oldest = min(idle, key=lambda w: w.last_active)
        self.forget(oldest)
        await oldest.shutdown()
        return True

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(min(30.0, settings.server.worker_idle_seconds))
            now = time.monotonic()
            for worker in list(self.workers.values()):
                if not worker.busy and now - worker.last_active >= settings.server.worker_idle_seconds:
                    self.forget(worker)
                    await worker.shutdown()
