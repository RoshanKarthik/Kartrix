"""The JSON-lines protocol between the server and a workspace's worker process.

Server → worker (stdin), one object per line, discriminated by ``op``:
``run`` · ``approval`` · ``plan_review`` · ``command`` · ``stop`` · ``shutdown``.

Worker → server (stdout), discriminated by ``kind``:
``ready`` · ``start_failed`` · ``event`` (a core event) · ``approval_needed`` · ``plan_review_needed`` ·
``run_finished`` · ``command_result`` (``request_id``, ``ok``, ``result`` or ``error``).

The same message kinds (minus ``ready``/``start_failed``/``command_result``) are what clients receive on a run's
SSE feed; a command's result is the answer to its HTTP request,
so a worker in a cloud sandbox (phase 3) only has to speak this protocol.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class Decision(BaseModel):
    """An answer to one approval request (mirrors :class:`kartrix.security.approvals.ApprovalDecision`)."""

    type: Literal["approve", "approve_session", "edit", "reject"]
    command: str | None = Field(None, max_length=10_000)  # "edit": the command to run instead
    message: str | None = Field(None, max_length=2_000)  # "reject": optional reason for the agent


class RunOp(BaseModel):
    op: Literal["run"] = "run"
    run_id: str
    kind: Literal["ask", "plan"]
    input: str
    session_id: str
    model: str | None = None  # catalog id of the main model (None: the default)


class ApprovalOp(BaseModel):
    op: Literal["approval"] = "approval"
    request_id: str
    decisions: list[Decision]


class PlanReviewOp(BaseModel):
    op: Literal["plan_review"] = "plan_review"
    request_id: str
    approved: bool
    plan: dict[str, Any] | None = None  # the plan as approved (maybe edited); None: as proposed
    feedback: str = ""  # approved=False: what should change in the re-plan


class CommandOp(BaseModel):
    """A workspace command (:mod:`kartrix.core.commands`): ``/undo``, ``/mode``, ``/memory`` … The worker
    answers with ``command_result`` (same ``request_id``), never on a run's feed."""

    op: Literal["command"] = "command"
    request_id: str
    name: str = Field(max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)


class StopOp(BaseModel):
    op: Literal["stop"] = "stop"


class ShutdownOp(BaseModel):
    op: Literal["shutdown"] = "shutdown"


Op = Annotated[RunOp | ApprovalOp | PlanReviewOp | CommandOp | StopOp | ShutdownOp, Field(discriminator="op")]
OP_ADAPTER: TypeAdapter[Op] = TypeAdapter(Op)

# Message kinds a client sees on the SSE feed.
FEED_KINDS = ("event", "approval_needed", "plan_review_needed", "run_finished")


def encode(message: BaseModel | dict[str, Any]) -> str:
    data = message.model_dump(mode="json") if isinstance(message, BaseModel) else message
    return json.dumps(data, ensure_ascii=False, default=str) + "\n"


def decode(line: str | bytes) -> dict[str, Any]:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("a protocol message must be a JSON object")
    return data
