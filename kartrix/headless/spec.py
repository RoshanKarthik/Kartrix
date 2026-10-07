"""The run spec for ``kartrix run --spec <file>`` (YAML, loaded strictly: unknown or duplicate keys are errors).

Example::

    task: Add a /health endpoint with a test
    mode: plan                  # ask (one chat request) | plan (plan → execute)
    permissions: auto           # read_only | default | auto
    budget: {max_cost_usd: 1.0, max_seconds: 900}   # on top of budgets.turn / budgets.plan
    approvals:                  # answers the approval requests nobody is there to answer
      allow: ["npm install *", "pytest *"]
      deny: ["curl *"]
      allow_tools: []           # MCP tools that need approval, by name
      otherwise: reject         # unmatched requests are declined (the agent is told)
    plan_only: false            # plan mode: stop after saving the (auto-approved) plan
    report: report.json         # default: printed to stdout
    events: events.jsonl        # every event as one JSON line
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kartrix.config import BudgetLimits, ConfigError, load_yaml_strict


class SpecError(Exception):
    """The spec file is missing or invalid. Safe to show to the user."""


class ApprovalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Command rules as in permissions.allow: "npm run *" (trailing * = any arguments), "make test".
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)  # wins over allow
    allow_tools: list[str] = Field(default_factory=list)  # MCP tools needing approval, by exact name
    otherwise: Literal["reject"] = "reject"


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    mode: Literal["ask", "plan"] = "ask"
    permissions: Literal["read_only", "default", "auto"] = "default"
    budget: BudgetLimits | None = None  # fields given here replace the configured ones for this run
    approvals: ApprovalPolicy = ApprovalPolicy()
    plan_only: bool = False
    semantic_cache: bool = False  # off by default: a cached answer would skip the agent being measured
    mcp: list[str] = Field(default_factory=list)  # MCP servers to connect for this run
    report: str | None = None
    events: str | None = None

    @field_validator("task")
    @classmethod
    def _task_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task must not be empty")
        return v.strip()


def load_spec(path: str | Path) -> RunSpec:
    p = Path(path)
    if not p.is_file():
        raise SpecError(f"spec file not found: {p}")
    try:
        data = load_yaml_strict(p)
    except ConfigError as e:
        raise SpecError(str(e)) from None
    try:
        return RunSpec.model_validate(data)
    except ValidationError as e:
        raise SpecError(f"invalid spec {p}:\n{e}") from None
