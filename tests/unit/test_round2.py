"""Router override, plan validation and rate-limit-aware retries."""

from __future__ import annotations

from typing import Any

import pytest

from kartrix.agent.graph import looks_like_change
from kartrix.llm import model_retry
from kartrix.tasks.planner import ExecutionPlan, PlannedTask, plan_problems


@pytest.mark.parametrize(
    "request_text",
    [
        'Add a --json flag to "notes stats" that prints one JSON object',
        "fix the pagination bug",
        "Please rename TodoStore to TaskStore",
        "Can you write tests for slugify?",
        "Refactor the storage layer",
    ],
)
def test_imperative_requests_are_changes(request_text: str) -> None:
    assert looks_like_change(request_text)


@pytest.mark.parametrize(
    "request_text",
    ["Where is the config loaded?", "What does add do?", "why does the address field fail?", "Explain the fix-up"],
)
def test_questions_stay_questions(request_text: str) -> None:
    assert not looks_like_change(request_text)


def _task(i: int, deps: list[str] | None = None, title: str = "Add search command") -> PlannedTask:
    return PlannedTask(
        id=f"task_{i:03}", title=title, description="do it", task_type="implement", depends_on=deps or [],
        estimated_minutes=5, output_files=["notes/cli.py"], acceptance_criteria=["pytest passes"],
    )  # fmt: skip


def _plan(*tasks: PlannedTask, name: str = "notes search") -> ExecutionPlan:
    return ExecutionPlan(
        project_name=name, goal_summary="add search", tech_stack=["Python"], total_estimated_hours=0.5,
        tasks=list(tasks), risks=[], assumptions=[],
    )  # fmt: skip


def test_a_good_plan_has_no_problems() -> None:
    assert plan_problems(_plan(_task(1), _task(2, ["task_001"]))) == []


def test_placeholder_plans_unknown_deps_and_cycles_are_rejected() -> None:
    assert any("placeholder" in p for p in plan_problems(_plan(_task(1, title="Test task"), name="test")))
    assert any("unknown" in p for p in plan_problems(_plan(_task(1, ["task_009"]))))
    cycle = plan_problems(_plan(_task(1, ["task_002"]), _task(2, ["task_001"])))
    assert any("cycle" in p for p in cycle)
    assert plan_problems(_plan()) == ["the plan has no tasks"]


class _RateLimited(Exception):
    def __init__(self, retry_after: str | None = None) -> None:
        super().__init__("[429] Too Many Requests")
        self.status_code = 429
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


def test_rate_limits_wait_longer_and_honour_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = model_retry.settings.llm.retry
    monkeypatch.setattr(policy, "rate_limit_retries", 3)
    monkeypatch.setattr(policy, "rate_limit_max_delay", 60.0)
    assert 7.0 <= (model_retry.delay_for(_RateLimited("7"), 0) or 0) <= 8.0
    assert 2.5 <= (model_retry.delay_for(_RateLimited(), 0) or 0) <= 5.0
    assert 10.0 <= (model_retry.delay_for(_RateLimited(), 2) or 0) <= 20.0
    assert model_retry.delay_for(_RateLimited(), 3) is None  # out of retries
    assert model_retry.delay_for(ValueError("bad request"), 0) is None  # not transient


async def test_retry_middleware_retries_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_retry, "delay_for", lambda exc, attempt: 0.0 if attempt < 2 else None)
    calls: list[int] = []

    async def handler(request: Any) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _RateLimited()
        return "ok"

    assert await model_retry.RateAwareRetryMiddleware().awrap_model_call(None, handler) == "ok"  # type: ignore[arg-type]
    assert len(calls) == 3
