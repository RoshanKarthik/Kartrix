import asyncio
import uuid

import pytest
from pydantic import BaseModel

from kartrix.db.engine import session_scope
from kartrix.db.models import Session
from kartrix.tasks.recovery import RecoveryManager
from kartrix.tasks.task_store import ProjectStatus, TaskStore, TaskType

pytestmark = pytest.mark.usefixtures("db")

REPO = "D:/fake/repo"


class PlanTask(BaseModel):
    id: str
    title: str
    description: str = "-"
    task_type: TaskType
    depends_on: list[str] = []
    output_files: list[str] = []
    acceptance_criteria: list[str] = []


class Plan(BaseModel):
    project_name: str
    tasks: list[PlanTask]


PLAN = Plan(
    project_name="demo",
    tasks=[
        PlanTask(id="task_1", title="design", task_type=TaskType.DESIGN),
        PlanTask(id="task_2", title="impl", task_type=TaskType.IMPLEMENT, depends_on=["task_1"]),
        PlanTask(id="task_3", title="test", task_type=TaskType.TEST, depends_on=["task_1", "task_2"]),
    ],
)


async def _ready(store: TaskStore, pid: str) -> list[str]:
    return [t["id"] for t in await store.get_ready_tasks(pid)]


async def test_dependencies_gate_readiness() -> None:
    store = TaskStore()
    pid = await store.create_project("goal", PLAN, REPO)  # type: ignore[arg-type]
    assert await _ready(store, pid) == ["task_1"]
    assert await store.claim_task(pid, "task_1")
    assert await _ready(store, pid) == []
    await store.complete_task(pid, "task_1", "done1")
    assert await _ready(store, pid) == ["task_2"]
    assert await store.get_dep_results(pid, ["task_1", "task_2"]) == [
        {"id": "task_1", "title": "design", "result": "done1"}
    ]


async def test_same_keys_in_two_projects_do_not_clash() -> None:
    # Regression: planner keys used to be the global primary key.
    store = TaskStore()
    p1 = await store.create_project("g1", PLAN, REPO)  # type: ignore[arg-type]
    p2 = await store.create_project("g2", PLAN, REPO)  # type: ignore[arg-type]
    assert await store.claim_task(p1, "task_1")
    await store.complete_task(p1, "task_1", "r")
    assert await _ready(store, p1) == ["task_2"]
    assert await _ready(store, p2) == ["task_1"]


async def test_only_one_concurrent_claim_wins() -> None:
    store = TaskStore()
    pid = await store.create_project("g", PLAN, REPO)  # type: ignore[arg-type]
    wins = await asyncio.gather(*[store.claim_task(pid, "task_1") for _ in range(5)])
    assert sum(wins) == 1


async def test_failures_retry_then_fail_and_block_dependents() -> None:
    store = TaskStore()
    pid = await store.create_project("g", PLAN, REPO)  # type: ignore[arg-type]
    await store.claim_task(pid, "task_1")
    await store.complete_task(pid, "task_1", "ok")
    statuses = []
    for _ in range(3):
        assert await store.claim_task(pid, "task_2")
        await store.fail_task(pid, "task_2", "boom")
        statuses.append({t["id"]: t for t in await store.get_all_tasks(pid)}["task_2"]["status"])
    assert statuses == ["pending", "pending", "failed"]
    assert await _ready(store, pid) == []  # task_3 waits on a failed dependency
    assert await store.get_progress(pid) == {"completed": 1, "failed": 1, "pending": 1}


async def test_crash_recovery() -> None:
    store = TaskStore()
    pid = await store.create_project("g", PLAN, REPO)  # type: ignore[arg-type]
    await store.claim_task(pid, "task_1")  # process "dies" here
    assert await RecoveryManager(store).recover(pid) == 1
    t1 = (await store.get_all_tasks(pid))[0]
    assert (t1["status"], t1["retry_count"], t1["error"]) == ("pending", 1, "CRASH: process died mid-execution")


async def test_resume_picks_unfinished_project_of_this_repo() -> None:
    # Regression: projects stayed "approved" forever, so /plan always resumed a finished one.
    store = TaskStore()
    p1 = await store.create_project("g1", PLAN, REPO)  # type: ignore[arg-type]
    await store.create_project("other repo", PLAN, "D:/elsewhere")  # type: ignore[arg-type]
    assert await store.get_resumable_project(REPO) == p1
    await store.set_project_status(p1, ProjectStatus.COMPLETED)
    assert await store.get_resumable_project(REPO) is None
    assert await store.get_latest_project(REPO) == p1


async def test_project_links_to_session_and_lists_are_lists() -> None:
    sid = str(uuid.uuid4())
    async with session_scope() as s:
        s.add(Session(id=uuid.UUID(sid), repo_path=REPO))
    store = TaskStore()
    pid = await store.create_project("g", PLAN, REPO, sid)  # type: ignore[arg-type]
    t3 = (await store.get_all_tasks(pid))[2]
    assert t3["depends_on"] == ["task_1", "task_2"]
    assert t3["project_id"] == pid
