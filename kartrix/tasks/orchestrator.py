from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kartrix.context.indexers.pg_index import index_repo
from kartrix.core.events import (
    PlanProposed,
    PlanReviewed,
    Progress,
    ProjectFinished,
    ProjectStarted,
    TaskFinished,
    TaskStarted,
    emit,
    notice,
)
from kartrix.core.interaction import PlanReviewer
from kartrix.observability.logger import get_logger
from kartrix.security import checkpoints
from kartrix.security.approvals import Approver
from kartrix.security.audit import audit_scope
from kartrix.security.budget import RunStopped, raise_if_stopped, waiting_for_user
from kartrix.tasks.executor import run_subtask_agent
from kartrix.tasks.planner import ExecutionPlan, create_plan, planning_context
from kartrix.tasks.recovery import RecoveryManager
from kartrix.tasks.task_store import ProjectStatus, TaskStore

logger = get_logger(__name__)

ProjectOutcome = Literal["completed", "failed", "stopped", "planned"]


@dataclass
class PlanResult:
    """What a /plan run did — for the REPL's summary and the headless report."""

    project_id: str | None = None
    status: ProjectOutcome | Literal["not_started"] = "not_started"
    resumed: bool = False
    plan: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None


class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    Serial by default (max_concurrent=1) for cost control and determinism.
    """

    def __init__(
        self,
        store: TaskStore,
        max_concurrent: int = 1,
        approver: Approver | None = None,
        session_id: str | None = None,
    ) -> None:
        self.store = store
        self.max_concurrent = max_concurrent
        self.approver = approver
        self.session_id = session_id

    async def run(self, project_id: str) -> ProjectOutcome:
        """
        Loop until all tasks reach a terminal state.

        Each iteration:
          1. Check progress — exit if nothing pending or in-progress
          2. Get ready tasks (all deps completed/skipped)
          3. Dispatch up to max_concurrent tasks
          4. Sleep 5s and repeat if waiting on in-progress tasks
        """
        await self.store.set_project_status(project_id, ProjectStatus.RUNNING)

        while True:
            raise_if_stopped()  # budget used up or kill switch: the project stays resumable
            progress = await self.store.get_progress(project_id)
            pending = progress.get("pending", 0)
            in_progress = progress.get("in_progress", 0)
            emit(
                Progress(
                    completed=progress.get("completed", 0),
                    total=sum(progress.values()),
                    in_progress=in_progress,
                    pending=pending,
                    failed=progress.get("failed", 0),
                    blocked=progress.get("blocked", 0),
                )
            )
            # all tasks are done - nothing in pending and nothing is in progress
            if pending == 0 and in_progress == 0:
                ok = progress.get("failed", 0) == 0 and progress.get("blocked", 0) == 0
                await self.store.set_project_status(project_id, ProjectStatus.COMPLETED if ok else ProjectStatus.FAILED)
                outcome: ProjectOutcome = "completed" if ok else "failed"
                emit(ProjectFinished(project_id=project_id, status=outcome, counts=progress))
                return outcome

            ready = await self.store.get_ready_tasks(project_id)
            if not ready:
                if in_progress == 0:
                    notice("No tasks are ready — some are blocked by failed dependencies.", "warning")
                    await self.store.set_project_status(project_id, ProjectStatus.FAILED)
                    emit(ProjectFinished(project_id=project_id, status="failed", counts=progress))
                    return "failed"
                await asyncio.sleep(5)
                continue

            batch = ready[: self.max_concurrent]
            await asyncio.gather(*[self._execute(project_id, task) for task in batch])

    async def _execute(self, project_id: str, task: dict) -> None:
        """Claim and execute a single task, handling retries via fail_task."""
        key, title = task["id"], task["title"]
        if not await self.store.claim_task(project_id, key):
            logger.warning(f"Task {key} already claimed — skipping")
            return
        emit(TaskStarted(task_key=key, title=title))

        try:
            # Fetch what dependency tasks actually produced and inject into the agent.
            dep_outputs = await self.store.get_dep_results(project_id, task["depends_on"])

            # A checkpoint before each task: /undo reverts the files it changed.
            async with checkpoints.track(f"task {key}: {title}", self.session_id):
                with audit_scope(session_id=self.session_id, project_id=project_id, task_key=key):
                    result = await run_subtask_agent(task, dep_outputs=dep_outputs, approver=self.approver)
            await self.store.complete_task(project_id, key, result)
            emit(TaskFinished(task_key=key, title=title, status="completed"))

        except (RunStopped, asyncio.CancelledError) as e:
            # Stopped, not failed: the task goes back to pending without using a retry.
            reason = e.reason if isinstance(e, RunStopped) else "stopped by the kill switch"
            await self.store.release_task(project_id, key, reason)
            emit(TaskFinished(task_key=key, title=title, status="stopped", detail=reason))
            raise

        except Exception as e:
            error_msg = str(e)
            await self.store.fail_task(project_id, key, error_msg)
            emit(TaskFinished(task_key=key, title=title, status="failed", detail=error_msg[:500]))
            logger.error(f"Task {key} failed: {error_msg}")


async def _plan(goal: str, reviewer: PlanReviewer) -> ExecutionPlan:
    """Plan → review → re-plan with feedback until the reviewer approves."""
    extra_context = ""
    repo_context = await planning_context(goal)  # files, repo map and the code relevant to the goal
    while True:
        notice("Planning (this may take a moment)...")
        raw_plan = await asyncio.to_thread(create_plan, goal, extra_context, repo_context)
        emit(PlanProposed(plan=raw_plan.model_dump(mode="json")))
        with waiting_for_user():  # reading the plan doesn't use the time budget
            review = await reviewer(raw_plan)
        raise_if_stopped()
        emit(PlanReviewed(approved=review.plan is not None, by=review.by, feedback=review.feedback or None))
        if review.plan is not None:
            return review.plan
        extra_context = review.feedback.strip()


async def handle_plan_command(
    goal: str,
    session_id: str | None = None,
    approver: Approver | None = None,
    reviewer: PlanReviewer | None = None,
    *,
    resume: Literal["any", "same_goal"] = "any",
    plan_only: bool = False,
) -> PlanResult:
    """
    Full /plan flow:

      1. An unfinished (approved/running) project of this repo → resume + recover it
         (``resume="same_goal"``: only one with exactly this goal — headless runs)
      2. Otherwise: plan → review loop → persist → execute (``plan_only``: stop after saving)

    ``approver`` answers the tasks' command approvals (without one they are refused);
    ``reviewer`` approves the plan (without one it is approved as proposed).
    """
    from kartrix.core.interaction import approve_plan_as_is

    store = TaskStore()
    repo_path = str(Path.cwd().resolve())
    result = PlanResult()

    project_id = await store.get_resumable_project(repo_path, goal if resume == "same_goal" else None)
    if project_id:
        result.resumed = True
        recovered = await RecoveryManager(store).recover(project_id)
        emit(ProjectStarted(project_id=project_id, resumed=True, recovered=recovered))
    else:
        try:
            plan = await _plan(goal, reviewer or approve_plan_as_is)
        except RunStopped as e:
            notice(f"Planning stopped: {e.reason}. Nothing was saved.", "warning")
            result.status, result.stop_reason = "stopped", e.reason
            return result
        result.plan = plan.model_dump(mode="json")
        project_id = await store.create_project(goal, plan, repo_path, session_id)
        emit(ProjectStarted(project_id=project_id, resumed=False))
    result.project_id = project_id

    if plan_only:
        result.status = "planned"
        result.tasks = await store.get_all_tasks(project_id)
        emit(ProjectFinished(project_id=project_id, status="planned"))
        return result

    orchestrator = TaskOrchestrator(store, max_concurrent=1, approver=approver, session_id=session_id)
    try:
        result.status = await orchestrator.run(project_id)
    except RunStopped as e:
        result.status, result.stop_reason = "stopped", e.reason
        emit(ProjectFinished(project_id=project_id, status="stopped", counts=await store.get_progress(project_id)))
    result.tasks = await store.get_all_tasks(project_id)
    if result.status == "stopped":
        return result

    # Re-index so follow-up questions can find the files the tasks just generated.
    notice("Re-indexing generated files...")
    try:
        await index_repo(Path.cwd())
        notice("Index updated — questions can now find the generated code.", "success")
    except Exception as e:
        logger.warning(f"Re-index after /plan failed: {e}")
        notice(f"Re-index failed: {e}", "warning")
    return result
