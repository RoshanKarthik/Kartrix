"""What the core may ask the person (or policy) driving it — the only way back from core to front end.

- :data:`~kartrix.security.approvals.Approver` answers command/tool approvals.
- :data:`PlanReviewer` reviews a proposed plan: approve it (possibly edited) or ask for a re-plan.

The REPL answers with terminal prompts (``kartrix.ui``); headless mode answers from the run's
spec (``kartrix.headless``). Neither half imports the other.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from kartrix.security.approvals import ApprovalDecision, ApprovalRequest, Approver
from kartrix.tasks.planner import ExecutionPlan


@dataclass(frozen=True)
class PlanReview:
    plan: ExecutionPlan | None  # the approved plan (maybe edited); None = re-plan
    feedback: str = ""  # what should change in the re-plan
    by: str = "user"


PlanReviewer = Callable[[ExecutionPlan], Awaitable[PlanReview]]


async def approve_plan_as_is(plan: ExecutionPlan) -> PlanReview:
    """Reviewer for runs nobody watches: the plan is approved and recorded (headless)."""
    return PlanReview(plan, by="policy")


__all__ = ["ApprovalDecision", "ApprovalRequest", "Approver", "PlanReview", "PlanReviewer", "approve_plan_as_is"]
