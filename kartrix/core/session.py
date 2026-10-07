"""A Kartrix session without a terminal: start up in the current directory, run chat requests
and plans — each budgeted, stoppable (Ctrl+C / ``kartrix stop``) and checkpointed — and report
what happened as events plus a :class:`RunOutcome`.

The REPL (``kartrix.main``) and headless mode (``kartrix.headless``) both drive this class; they
differ only in how they answer questions (approver, plan reviewer) and render events.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kartrix.agent.factory import NATIVE_TOOLS, build_agent
from kartrix.agent.orchestrator import handle_query
from kartrix.cache.semantic_cache import build_semantic_cache, get_repo_domain
from kartrix.config import BudgetLimits, settings
from kartrix.context.index_status import set_status as set_index_status
from kartrix.context.indexers.pg_index import index_repo
from kartrix.context.indexers.watcher import start_watcher, stop_watcher
from kartrix.core.events import AssistantMessage, RunFinished, RunStarted, RunStatus, Usage, emit, notice, run_scope
from kartrix.core.interaction import PlanReviewer
from kartrix.db.engine import dispose_engine
from kartrix.llm.factory import get_embedder, get_llm
from kartrix.mcp.mcp_client import McpManager
from kartrix.memory.session import get_current_session, record_session
from kartrix.memory.short_term import get_checkpointer
from kartrix.observability.logger import get_logger
from kartrix.sandbox.manager import status as sandbox_status
from kartrix.security import checkpoints
from kartrix.security.approvals import Approver, resume_pending
from kartrix.security.budget import Budget, RunKind
from kartrix.security.kill_switch import close_dangling_tool_calls, run_stoppable
from kartrix.security.permissions import get_mode
from kartrix.security.workspace import set_workspace
from kartrix.skills.skill_tools import get_registry
from kartrix.tasks.orchestrator import PlanResult, handle_plan_command

logger = get_logger(__name__)


class _Clock:
    """Records how long each startup phase took (``CoreSession.startup_timings``)."""

    def __init__(self, into: dict[str, float]) -> None:
        self.into = into

    @contextlib.contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.into[name] = round(time.perf_counter() - start, 3)


def usage_of(budget: Budget) -> Usage:
    return Usage(
        input_tokens=budget.input_tokens,
        output_tokens=budget.output_tokens,
        estimated=budget.estimated,
        cost_usd=round(budget.cost_usd, 6),
        cost_known=not budget.unpriced,
        model_calls=budget.model_calls,
        tool_calls=budget.tool_calls,
        seconds=round(budget.elapsed(), 2),
    )


@dataclass
class RunOutcome:
    kind: Literal["ask", "plan"]
    status: RunStatus
    usage: Usage
    answer: str | None = None
    cached: bool = False
    plan: PlanResult | None = None
    detail: str | None = None  # stop reason or error
    files_changed: list[str] = field(default_factory=list)
    startup: dict[str, float] = field(default_factory=dict)  # seconds per startup phase (headless reports)


@dataclass
class StartOptions:
    session_id: str | None = None  # None: the workspace's current session (.kartrix/current_session)
    watch: bool = True  # re-index changed files in the background
    mcp: bool = True  # reconnect the MCP servers the user turned on
    semantic_cache: bool = True  # if enabled in config
    resume_pending: bool = True  # finish an approval the session was waiting for
    # background: start answering at once, the index catches up (search says so meanwhile);
    # wait: the index is up to date before start() returns (headless runs, the evals)
    index: Literal["background", "wait"] = "background"


class CoreSession:
    def __init__(self, approver: Approver, reviewer: PlanReviewer | None) -> None:
        self.approver = approver
        self.reviewer = reviewer
        self.checkpointer: Any = None
        self.agent: Any = None
        self.session_id = ""
        self.repo_path = str(Path.cwd())
        self.semantic_cache: Any = None
        self.cache_domain: str | None = None
        self.mcp: McpManager | None = None
        self.last_budget: Budget | None = None
        self.startup_timings: dict[str, float] = {}
        self._observer: Any = None
        self._index_task: asyncio.Task[None] | None = None
        self._watch = False
        self._stop_tracing: Any = None

    @classmethod
    async def start(
        cls, approver: Approver, reviewer: PlanReviewer | None = None, options: StartOptions | None = None
    ) -> CoreSession:
        """Bootstrap LLM, embedder, workspace jail, checkpoints, sandbox, index, cache, MCP, agent, session."""
        opts = options or StartOptions()
        self = cls(approver, reviewer)
        self._watch = opts.watch
        clock = _Clock(self.startup_timings)
        # Built once up front so a missing API key or bad provider config fails at startup.
        with clock("llm"):
            get_llm()
            get_embedder()
        notice(f"LLM: {settings.llm.provider} / {settings.llm.model}")
        notice(f"Embedder: {settings.embeddings.provider} / {settings.embeddings.model}")

        workspace = set_workspace(self.repo_path)  # file tools may only touch this tree
        notice(f"Workspace: {workspace.root} · permission mode: {get_mode()}")
        self._start_tracing()
        with clock("checkpoints"):
            off = checkpoints.setup()
            if off:
                notice(f"Checkpoints off: {off}", "warning")
            else:
                undo, _ = await asyncio.to_thread(checkpoints.current().entries)  # type: ignore[union-attr]
                notice(f"Checkpoints: on ({len(undo)} undo points) — /undo reverts the last change")
        with clock("sandbox"):
            sandbox = await asyncio.to_thread(sandbox_status)
        notice(f"Sandbox: {sandbox.describe()}", "info" if sandbox.backend is not None else "warning")

        with clock("semantic_cache"):
            if opts.semantic_cache:
                self.semantic_cache = await build_semantic_cache()
            self.cache_domain = get_repo_domain(self.repo_path) if self.semantic_cache else None
        if self.semantic_cache is not None:
            notice(f"Semantic cache: enabled (threshold={self.semantic_cache.threshold})")
        else:
            notice("Semantic cache: disabled")

        with clock("mcp"):
            self.mcp = McpManager(t.name for t in NATIVE_TOOLS)
            if opts.mcp:
                for name, error in (await self.mcp.connect_remembered()).items():
                    notice(f"MCP server {name} not connected: {error}", "warning")
                if self.mcp.connected:
                    notice(f"MCP: {', '.join(self.mcp.connected)} ({len(self.mcp.tools)} tools)")
        registry = get_registry()
        pending_skills = [s.name for s in registry.skills() if registry.status(s.name) != "trusted"]
        if pending_skills:
            notice(f"This repo has skills you haven't approved: {', '.join(pending_skills)} — see /skills", "warning")

        with clock("agent"):
            self.checkpointer = get_checkpointer()
            self.rebuild_agent()
        with clock("session"):
            self.session_id = opts.session_id or get_current_session()
            await record_session(self.session_id, self.repo_path)
        notice(f"Session: {self.session_id}")

        if opts.index == "wait":
            with clock("index"):
                await self._update_index()
        else:
            set_index_status("building")  # before the task runs: search must never see a stale "ready"
            self._index_task = asyncio.create_task(self._update_index(), name="kartrix-index")
        if opts.resume_pending:
            await self.finish_pending_approval()
        logger.info("Startup timing", extra={"seconds": self.startup_timings, "index": opts.index})
        return self

    # ── state ──────────────────────────────────────────────────────────

    def rebuild_agent(self) -> None:
        """After the tool set changed (MCP servers, trusted skills)."""
        self.agent = build_agent(self.checkpointer, self.mcp.tools if self.mcp else [])

    async def reindex(self) -> None:
        """Bring the index up to date now (``/reindex``)."""
        if self._index_task is not None and not self._index_task.done():
            await self._index_task  # the background update already does it
            return
        await self._update_index()

    async def _update_index(self) -> None:
        """Index the workspace; then watch it for changes (if enabled). Never raises."""
        set_index_status("building")
        start = time.perf_counter()
        try:
            stats = await index_repo(self.repo_path)
        except Exception as e:
            logger.error("Indexing failed", extra={"error": repr(e)})
            set_index_status("failed", f"{type(e).__name__}: {e}")
            notice(f"Indexing failed: {e} — search falls back to grep/glob", "warning")
            return
        set_index_status("ready", str(stats))
        self.startup_timings["index"] = round(time.perf_counter() - start, 2)
        notice(f"Index ready: {stats}")
        if self._watch and self._observer is None:
            self._observer = await asyncio.to_thread(
                start_watcher, self.repo_path, asyncio.get_running_loop(), self.invalidate_cache
            )

    @property
    def ready_for_search(self) -> bool:
        return self._index_task is None or self._index_task.done()

    async def invalidate_cache(self) -> None:
        if self.semantic_cache is not None and self.cache_domain is not None:
            await self.semantic_cache.invalidate_domain(self.cache_domain)

    async def use_session(self, session_id: str) -> None:
        self.session_id = session_id
        await record_session(session_id, self.repo_path)

    async def finish_pending_approval(self) -> None:
        """A session closed while asking for approval is still paused there: ask again and finish it."""
        config = {"configurable": {"thread_id": self.session_id}}
        try:
            state = await self.agent.aget_state(config)
            if not state.interrupts:
                return
            notice("This session stopped while waiting for your approval:", "warning")
            result = await resume_pending(self.agent, config, self.approver)
        except Exception as e:
            logger.error(f"Could not resume the pending approval: {e}")
            notice(f"Could not resume the pending approval: {e}", "error")
            return
        if result and result.get("messages"):
            emit(AssistantMessage(text=str(result["messages"][-1].content)))

    def _start_tracing(self) -> None:
        """A local trace of every run (``kartrix trace``); LangSmith too when its key is set."""
        from kartrix.core.events import bus
        from kartrix.observability.tracing import TraceRecorder, enable_langsmith

        if settings.tracing.enabled:
            recorder = TraceRecorder()
            unsubscribe = bus.subscribe(recorder)

            def stop() -> None:
                unsubscribe()
                recorder.close()

            self._stop_tracing = stop
        if enable_langsmith():
            notice(f"LangSmith tracing: on (project {settings.tracing.langsmith_project}) — prompts go to LangSmith")

    async def close(self) -> None:
        if self._stop_tracing is not None:
            self._stop_tracing()
            self._stop_tracing = None
        if self._index_task is not None and not self._index_task.done():
            self._index_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._index_task
        if self.mcp is not None:
            await self.mcp.close()  # same task that opened the sessions
        if self._observer is not None:
            stop_watcher(self._observer)
        await dispose_engine()

    # ── runs ───────────────────────────────────────────────────────────

    def _budget(self, kind: RunKind, limits: BudgetLimits | None) -> Budget:
        if limits is None:
            budget = Budget.for_run(kind)
        else:
            budget = Budget(kind, limits, settings.budgets.prices)
        self.last_budget = budget
        return budget

    async def _guarded(self, coro: Any, budget: Budget) -> Any:
        """Run inside ``budget``, stoppable; None if the kill switch cancelled it."""
        result = await run_stoppable(coro, budget, on_stop=lambda msg: notice(msg, "warning"))
        if budget.stop_reason:
            notice(f"Stopped: {budget.stop_reason} · {budget.summary()}", "warning")
            await budget.record_stop(self.session_id)
        return result

    async def ask(self, question: str, limits: BudgetLimits | None = None) -> RunOutcome:
        """One chat request: checkpointed (for /undo), budgeted and stoppable."""
        budget = self._budget("turn", limits)
        with run_scope(uuid.uuid4().hex[:12]):
            emit(RunStarted(kind="ask", input=question, session_id=self.session_id))
            outcome = RunOutcome("ask", "completed", Usage())
            try:
                async with checkpoints.track(f"/ask {question}", self.session_id) as tracked:
                    answer = await self._guarded(
                        handle_query(
                            self.agent, question, self.session_id, self.approver,
                            semantic_cache=self.semantic_cache, cache_domain=self.cache_domain,
                        ),
                        budget,
                    )  # fmt: skip
            except Exception as e:
                logger.error(f"Agent error: {e}")
                outcome.status, outcome.detail = "error", f"{type(e).__name__}: {e}"
                answer = None
            else:
                outcome.files_changed = list(tracked.entry.files) if tracked.entry else []
            if answer is None and outcome.status != "error":  # cancelled mid-step
                await self._close_dangling(budget)
            if answer is not None:
                outcome.answer, outcome.cached = answer.text, answer.cached
                emit(AssistantMessage(text=answer.text, cached=answer.cached))
                if not answer.text.strip() and outcome.status == "completed":
                    outcome.status, outcome.detail = "failed", "the model returned no answer"
            stopped = budget.exhausted()  # also the tool-call limit, which ends with a wrap-up message
            if stopped and outcome.status != "error":
                outcome.status, outcome.detail = "stopped", stopped
            outcome.usage = usage_of(budget)
            emit(RunFinished(kind="ask", status=outcome.status, detail=outcome.detail, usage=outcome.usage))
            return outcome

    async def _close_dangling(self, budget: Budget) -> None:
        """Answer the tool calls that never got a result, so the conversation stays valid."""
        config = {"configurable": {"thread_id": self.session_id}}
        try:
            await close_dangling_tool_calls(self.agent, config, budget.stop_reason or "stopped")
        except Exception as e:
            logger.error(f"Could not tidy up the stopped turn: {e}")

    async def plan(
        self,
        goal: str,
        limits: BudgetLimits | None = None,
        *,
        resume: Literal["any", "same_goal"] = "any",
        plan_only: bool = False,
    ) -> RunOutcome:
        """Plan (reviewed by ``self.reviewer``) and execute it — or resume this repo's unfinished plan."""
        budget = self._budget("plan", limits)
        with run_scope(uuid.uuid4().hex[:12]):
            emit(RunStarted(kind="plan", input=goal, session_id=self.session_id))
            outcome = RunOutcome("plan", "completed", Usage())
            try:
                result: PlanResult | None = await self._guarded(
                    handle_plan_command(
                        goal, self.session_id, self.approver, self.reviewer, resume=resume, plan_only=plan_only
                    ),
                    budget,
                )
            except Exception as e:
                logger.error(f"Plan failed: {e}")
                outcome.status, outcome.detail = "error", f"{type(e).__name__}: {e}"
                result = None
            outcome.plan = result
            if result is not None:
                if result.status == "failed":
                    outcome.status = "failed"
                elif result.status == "stopped":
                    outcome.status, outcome.detail = "stopped", result.stop_reason
            stopped = budget.exhausted()  # also the tool-call limit, which ends with a wrap-up message
            if stopped and outcome.status != "error":
                outcome.status, outcome.detail = "stopped", stopped
            outcome.usage = usage_of(budget)
            emit(RunFinished(kind="plan", status=outcome.status, detail=outcome.detail, usage=outcome.usage))
            return outcome
