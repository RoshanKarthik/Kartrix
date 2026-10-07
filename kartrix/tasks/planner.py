from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from pydantic import BaseModel

from kartrix.llm.factory import get_chat_model, get_model_middleware
from kartrix.observability.logger import get_logger
from kartrix.security.budget import BudgetMiddleware, raise_if_stopped
from kartrix.tasks.task_store import TaskType

logger = get_logger(__name__)


class PlannedTask(BaseModel):
    id: str  # stable snake_case e.g. task_001
    title: str
    description: str
    task_type: TaskType
    depends_on: list[str]  # list of task IDs that must complete first
    estimated_minutes: int
    output_files: list[str]  # files this task will create/modify
    acceptance_criteria: list[str]  # what "done" looks like for the judge


class ExecutionPlan(BaseModel):
    project_name: str
    goal_summary: str
    tech_stack: list[str]
    total_estimated_hours: float
    tasks: list[PlannedTask]
    risks: list[str]
    assumptions: list[str]


_SYSTEM_PROMPT = """\
You are a senior software architect. Given a goal for an existing repository (or an empty folder for a new
project), produce an ExecutionPlan of concrete tasks for an AI coding agent.

Rules:
- As few tasks as the goal really needs: 1–3 for a small feature or fix, up to 12 for a new application.
- Base the plan on the repository shown below: name the real files to change, follow its structure,
  language, test framework and conventions. Never invent placeholder names ("test", "Test task", "foo").
- Task IDs: task_001, task_002, ...; depends_on references IDs of this plan and forms no cycle.
- Every task that changes behaviour includes or updates the tests that prove it (or has a test task after it).
- output_files lists every file the task writes; acceptance_criteria are 2–5 concrete, checkable items
  (commands to run, outputs to expect).
- task_type is one of: design, implement, test, review, integrate, configure.
"""

_PLACEHOLDER_WORDS = frozenset({"test", "test task", "task", "todo", "example", "foo", "placeholder", "untitled"})


async def planning_context(goal: str, root: str | Path | None = None, max_files: int = 150) -> str:
    """What the planner should know about the repository: its files, its most-used symbols and the code
    most relevant to the goal. Best effort — an empty or unindexed folder gives a shorter context."""
    from kartrix.agent.context import repo_map_section
    from kartrix.context.discovery import discover_files
    from kartrix.context.retrievers import graph
    from kartrix.context.retrievers.pg_hybrid import retrieve

    base = Path(root or Path.cwd())
    parts: list[str] = []
    try:
        files = [p.relative_to(base).as_posix() for p in discover_files(base)]
    except OSError as e:
        logger.warning("Planning context: could not list files", extra={"error": str(e)})
        files = []
    if files:
        shown = "\n".join(files[:max_files]) + (f"\n… {len(files) - max_files} more" if len(files) > max_files else "")
        parts.append(f"## Files in the repository\n{shown}")
    else:
        parts.append("## Files in the repository\n(none yet — this is a new project)")
    try:
        section = repo_map_section(await graph.repo_map(base), 400)
        if section is not None:
            parts.append(section.render())
        hits = await retrieve(goal, k=6, repo_root=base)
        if hits:
            snippets = [
                f"### {h['source']}:{h['start_line']}-{h['end_line']} {h.get('name') or ''}\n{h['content'][:1200]}"
                for h in hits
            ]
            parts.append("## Code most relevant to the goal\n" + "\n\n".join(snippets))
    except Exception as e:  # no index yet, database down, … — plan with what we have
        logger.warning("Planning context: no index context", extra={"error": repr(e)})
    return "\n\n".join(parts)


def plan_problems(plan: ExecutionPlan) -> list[str]:
    """Why a plan can't be executed as it is (empty list: it can)."""
    problems: list[str] = []
    if not plan.tasks:
        problems.append("the plan has no tasks")
    ids = [t.id for t in plan.tasks]
    if len(set(ids)) != len(ids):
        problems.append("task ids are not unique")
    known = set(ids)
    for t in plan.tasks:
        missing = [d for d in t.depends_on if d not in known]
        if missing:
            problems.append(f"{t.id} depends on unknown tasks {missing}")
        if t.id in t.depends_on:
            problems.append(f"{t.id} depends on itself")
        if t.title.strip().lower() in _PLACEHOLDER_WORDS or not t.description.strip():
            problems.append(f"{t.id} has a placeholder title or no description")
    if (
        plan.project_name.strip().lower() in _PLACEHOLDER_WORDS
        or plan.goal_summary.strip().lower() in _PLACEHOLDER_WORDS
    ):
        problems.append("project_name / goal_summary are placeholders — describe the actual goal")
    deps = {t.id: set(t.depends_on) for t in plan.tasks}
    seen: set[str] = set()
    while True:  # Kahn: whatever can't be ordered is in a cycle
        ready = {i for i, d in deps.items() if i not in seen and d <= seen}
        if not ready:
            break
        seen |= ready
    if len(seen) < len(deps) and not any("unknown" in p for p in problems):
        problems.append(f"tasks {sorted(set(deps) - seen)} form a dependency cycle")
    return problems


def create_plan(goal: str, extra_context: str = "", repo_context: str = "") -> ExecutionPlan:
    """Call the LLM planner and return a structured ExecutionPlan. A plan that can't be executed
    (placeholders, unknown dependencies, cycles) is sent back once with the problems."""
    llm = get_chat_model("main", temperature=0)

    planner_agent = create_agent(
        llm,
        tools=[],
        system_prompt=_SYSTEM_PROMPT,
        response_format=ExecutionPlan,
        middleware=[*get_model_middleware(temperature=0), BudgetMiddleware()],
    )

    user_message = f"Goal: {goal}"
    if repo_context:
        user_message += f"\n\n# The repository\n{repo_context}"
    if extra_context:
        user_message += f"\n\nAdditional context / change requests:\n{extra_context}"

    messages: list[Any] = [{"role": "user", "content": user_message}]
    plan: ExecutionPlan | None = None
    for attempt in range(2):
        result = planner_agent.invoke({"messages": messages})
        raise_if_stopped()
        plan = result["structured_response"]
        problems = plan_problems(plan)
        if not problems:
            break
        logger.warning("Plan rejected by validation", extra={"attempt": attempt + 1, "problems": problems})
        messages = [
            *messages,
            {"role": "assistant", "content": plan.model_dump_json()},
            {
                "role": "user",
                "content": "That plan can't be executed: " + "; ".join(problems) + ". Produce a corrected plan.",
            },
        ]
    assert plan is not None  # noqa: S101 — the loop ran at least once
    logger.info(f"Plan created: {plan.project_name} with {len(plan.tasks)} tasks")
    return plan
