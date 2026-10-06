from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import Boolean, CursorResult, case, exists, func, select, update
from sqlalchemy.orm import aliased

from kartrix.db.engine import session_scope
from kartrix.db.models import Project, ProjectStatus, Task, TaskStatus
from kartrix.observability.logger import get_logger

if TYPE_CHECKING:
    from kartrix.tasks.planner import ExecutionPlan  # planner imports TaskType from here

logger = get_logger(__name__)

__all__ = ["ProjectStatus", "TaskStatus", "TaskStore", "TaskType"]

# A project in one of these states is picked up again by the next /plan.
_RESUMABLE = (ProjectStatus.APPROVED, ProjectStatus.RUNNING)


class TaskType(enum.StrEnum):
    DESIGN = "design"
    IMPLEMENT = "implement"
    TEST = "test"
    REVIEW = "review"
    INTEGRATE = "integrate"
    CONFIGURE = "configure"


def _task_dict(t: Task) -> dict[str, Any]:
    """Row → plain dict. ``id`` is the planner's key (``task_1``), unique within a project."""
    return {
        "id": t.key,
        "project_id": str(t.project_id),
        "title": t.title,
        "description": t.description,
        "task_type": t.task_type,
        "status": t.status.value,
        "depends_on": list(t.depends_on or []),
        "output_files": list(t.output_files or []),
        "acceptance_criteria": list(t.acceptance_criteria or []),
        "result": t.result,
        "error": t.error,
        "retry_count": t.retry_count,
        "max_retries": t.max_retries,
        "execution_order": t.execution_order,
    }


class TaskStore:
    """
    Authoritative persistence layer for all task state (Postgres, async).

    State machine:
        PENDING ──(claim)──► IN_PROGRESS ──(success)──► COMPLETED  (terminal)
                                    │
                                    ├──(failure, retries left)──► PENDING
                                    ├──(failure, no retries)────► FAILED
                                    └──(dep failed)─────────────► BLOCKED
        SKIPPED  (terminal — human explicitly skipped)

    Design rules:
    - Every state transition is a single atomic UPDATE guarded by the expected current
      status — write to the DB before acting on the new state.
    - Tasks are addressed by (project_id, key); keys are the planner's ids and are only
      unique within a project.
    - Execution is single-process and serial, so any task left IN_PROGRESS when a project
      is resumed is unconditionally orphaned (see ``recover_crashed``).
    """

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    async def create_project(
        self, goal: str, plan: ExecutionPlan, repo_path: str, session_id: str | None = None
    ) -> str:
        """Persist an approved ExecutionPlan as a project + task rows. Returns the project id."""
        async with session_scope() as s:
            project = Project(
                name=plan.project_name[:200],
                goal=goal,
                repo_path=repo_path,
                plan=plan.model_dump(mode="json"),
                status=ProjectStatus.APPROVED,
                approved_at=func.now(),
                session_id=uuid.UUID(session_id) if session_id else None,
            )
            project.tasks = [
                Task(
                    key=pt.id,
                    title=pt.title[:300],
                    description=pt.description,
                    task_type=pt.task_type.value,
                    depends_on=list(pt.depends_on),
                    output_files=list(pt.output_files),
                    acceptance_criteria=list(pt.acceptance_criteria),
                    execution_order=i,
                )
                for i, pt in enumerate(plan.tasks)
            ]
            s.add(project)
            await s.flush()
            project_id = str(project.id)
        logger.info(f"Project {project_id} created with {len(plan.tasks)} tasks")
        return project_id

    async def get_resumable_project(self, repo_path: str) -> str | None:
        """Most recent approved/running project for this repo, or None."""
        async with session_scope() as s:
            pid = (
                await s.execute(
                    select(Project.id)
                    .where(Project.repo_path == repo_path, Project.status.in_(_RESUMABLE))
                    .order_by(Project.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return str(pid) if pid else None

    async def get_latest_project(self, repo_path: str) -> str | None:
        """Most recent project for this repo in any state (for /task_status)."""
        async with session_scope() as s:
            pid = (
                await s.execute(
                    select(Project.id)
                    .where(Project.repo_path == repo_path)
                    .order_by(Project.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return str(pid) if pid else None

    async def set_project_status(self, project_id: str, status: ProjectStatus) -> None:
        async with session_scope() as s:
            await s.execute(update(Project).where(Project.id == uuid.UUID(project_id)).values(status=status))

    # ------------------------------------------------------------------
    # Atomic state transitions
    # ------------------------------------------------------------------

    async def claim_task(self, project_id: str, key: str) -> bool:
        """Atomically PENDING → IN_PROGRESS. False if someone else already claimed it."""
        async with session_scope() as s:
            res = cast(
                CursorResult[Any],
                await s.execute(
                    update(Task)
                    .where(Task.project_id == uuid.UUID(project_id), Task.key == key, Task.status == TaskStatus.PENDING)
                    .values(status=TaskStatus.IN_PROGRESS, started_at=func.now())
                ),
            )
        return res.rowcount == 1

    async def complete_task(self, project_id: str, key: str, result: str) -> None:
        async with session_scope() as s:
            await s.execute(
                update(Task)
                .where(Task.project_id == uuid.UUID(project_id), Task.key == key)
                .values(status=TaskStatus.COMPLETED, result=result, completed_at=func.now())
            )
        logger.info(f"Task {key} completed")

    async def fail_task(self, project_id: str, key: str, error: str) -> None:
        """Record a failure; retries left → PENDING, exhausted → FAILED (one atomic UPDATE)."""
        async with session_scope() as s:
            await s.execute(
                update(Task)
                .where(Task.project_id == uuid.UUID(project_id), Task.key == key)
                .values(
                    retry_count=Task.retry_count + 1,
                    error=error,
                    status=case(
                        (Task.retry_count + 1 < Task.max_retries, TaskStatus.PENDING.value),
                        else_=TaskStatus.FAILED.value,
                    ),
                    started_at=None,
                )
            )
        logger.info(f"Task {key} failed: {error[:80]}")

    async def block_task(self, project_id: str, key: str, reason: str) -> None:
        async with session_scope() as s:
            await s.execute(
                update(Task)
                .where(Task.project_id == uuid.UUID(project_id), Task.key == key)
                .values(status=TaskStatus.BLOCKED, error=reason)
            )

    async def recover_crashed(self, project_id: str) -> list[dict[str, Any]]:
        """Reset tasks left IN_PROGRESS by a crashed run.

        Retries left → PENDING (counts as a retry), otherwise FAILED. Returns the
        affected tasks with their new status.
        """
        async with session_scope() as s:
            rows = (
                await s.execute(
                    update(Task)
                    .where(Task.project_id == uuid.UUID(project_id), Task.status == TaskStatus.IN_PROGRESS)
                    .values(
                        status=case(
                            (Task.retry_count < Task.max_retries, TaskStatus.PENDING.value),
                            else_=TaskStatus.FAILED.value,
                        ),
                        retry_count=case(
                            (Task.retry_count < Task.max_retries, Task.retry_count + 1), else_=Task.retry_count
                        ),
                        error=case(
                            (Task.retry_count < Task.max_retries, "CRASH: process died mid-execution"),
                            else_="CRASH: max retries exceeded after repeated crashes",
                        ),
                        started_at=None,
                    )
                    .returning(Task.key, Task.title, Task.status, Task.retry_count, Task.max_retries)
                )
            ).all()
        return [r._asdict() for r in rows]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_ready_tasks(self, project_id: str) -> list[dict[str, Any]]:
        """PENDING tasks of the project whose every dependency is COMPLETED or SKIPPED."""
        d = aliased(Task)
        unfinished_dep = exists().where(
            d.project_id == Task.project_id,
            func.jsonb_exists(Task.depends_on, d.key, type_=Boolean),
            d.status.not_in([TaskStatus.COMPLETED, TaskStatus.SKIPPED]),
        )
        async with session_scope() as s:
            rows = (
                (
                    await s.execute(
                        select(Task)
                        .where(
                            Task.project_id == uuid.UUID(project_id), Task.status == TaskStatus.PENDING, ~unfinished_dep
                        )
                        .order_by(Task.execution_order)
                    )
                )
                .scalars()
                .all()
            )
        return [_task_dict(t) for t in rows]

    async def get_all_tasks(self, project_id: str) -> list[dict[str, Any]]:
        async with session_scope() as s:
            rows = (
                (
                    await s.execute(
                        select(Task).where(Task.project_id == uuid.UUID(project_id)).order_by(Task.execution_order)
                    )
                )
                .scalars()
                .all()
            )
        return [_task_dict(t) for t in rows]

    async def get_progress(self, project_id: str) -> dict[str, int]:
        async with session_scope() as s:
            rows = (
                await s.execute(
                    select(Task.status, func.count())
                    .where(Task.project_id == uuid.UUID(project_id))
                    .group_by(Task.status)
                )
            ).all()
        return {status.value: n for status, n in rows}

    async def get_dep_results(self, project_id: str, dep_keys: list[str]) -> list[dict[str, Any]]:
        """Title + result of completed dependencies — injected into each subtask agent's prompt."""
        if not dep_keys:
            return []
        async with session_scope() as s:
            rows = (
                await s.execute(
                    select(Task.key, Task.title, Task.result).where(
                        Task.project_id == uuid.UUID(project_id),
                        Task.key.in_(dep_keys),
                        Task.status == TaskStatus.COMPLETED,
                    )
                )
            ).all()
        return [{"id": r.key, "title": r.title, "result": r.result} for r in rows]
