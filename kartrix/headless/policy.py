"""Answers approval requests from the run spec's ``approvals`` section (nobody is there to ask).

- a command matching a ``deny`` rule → declined; matching an ``allow`` rule → approved once;
- an MCP tool call → approved if its name is in ``allow_tools``;
- anything else → declined (``otherwise: reject``): the agent is told and can try another way.

Approval only ever applies to "ask" decisions: the command policy's hard deny list and deny
decisions still win, exactly as for a person answering in the terminal.
"""

from __future__ import annotations

import shlex

from kartrix.headless.spec import ApprovalPolicy
from kartrix.observability.logger import get_logger
from kartrix.security.approvals import ApprovalDecision, ApprovalRequest
from kartrix.security.command_policy import rule_matches

logger = get_logger(__name__)


def _argv(command: str) -> list[str] | None:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return None


class PolicyApprover:
    def __init__(self, policy: ApprovalPolicy) -> None:
        self.policy = policy

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.tool != "run_command":
            if request.tool in self.policy.allow_tools:
                return ApprovalDecision("approve", by="policy")
            return ApprovalDecision(
                "reject", message=f"{request.tool} is not in approvals.allow_tools of this run", by="policy"
            )
        argv = _argv(request.command)
        if argv:
            if any(rule_matches(rule, argv) for rule in self.policy.deny):
                return ApprovalDecision("reject", message="denied by this run's approval policy", by="policy")
            if any(rule_matches(rule, argv) for rule in self.policy.allow):
                return ApprovalDecision("approve", by="policy")
        return ApprovalDecision(
            "reject",
            message="not allowed by this run's approval policy (no approvals.allow rule matches) — "
            "nobody can approve it in this headless run; find another way or report what is needed",
            by="policy",
        )

    async def __call__(self, requests: list[ApprovalRequest]) -> list[ApprovalDecision]:
        decisions = [self.decide(r) for r in requests]
        logger.info(
            "Headless approval decisions",
            extra={"commands": [r.command for r in requests], "decisions": [d.type for d in decisions]},
        )
        return decisions
