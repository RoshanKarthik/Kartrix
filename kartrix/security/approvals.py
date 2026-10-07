"""Human-in-the-loop approvals (B3) for "ask" decisions of the command policy.

Two halves, joined by a LangGraph interrupt:

- :class:`ApprovalMiddleware` (inside the agent graph). After each model turn it evaluates
  the command policy for every ``run_command`` call; calls that need approval are batched
  into one ``interrupt()`` **before any tool of that turn runs**. The graph pauses and its
  state is checkpointed, so a pending request survives a restart. On resume, each call gets
  its decision by tool-call id: approved calls run with :func:`permissions.user_approved`
  set (the tool re-evaluates the policy, so a deny still wins), edited calls run the user's
  command instead, declined calls get an error result and never run.
- :func:`run_agent` / :func:`resume_pending` (the client). They drive the agent, hand
  interrupts to an :data:`Approver` (the CLI prompt today, the dashboard later), record
  each request in the ``approvals`` table and each decision in the audit log, and resume.

Decisions are keyed by tool-call id, not position (as in LangChain's
``HumanInTheLoopMiddleware``): when the graph resumes, the policy is evaluated again, and a
call may no longer need approval (e.g. the user just allowed that command for the session).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, PrivateStateAttr, ToolCallRequest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import Command, Interrupt, interrupt
from sqlalchemy import update

from kartrix.db.engine import session_scope
from kartrix.db.models import Approval, ApprovalKind, ApprovalStatus
from kartrix.observability.logger import get_logger
from kartrix.security import audit, external_tools, permissions
from kartrix.security.command_policy import Decision, evaluate
from kartrix.security.command_rules import Category
from kartrix.security.injection import external_content_cap
from kartrix.security.secrets import redact

logger = get_logger(__name__)

INTERRUPT_TYPE = "kartrix.approval"
DecisionType = Literal["approve", "approve_session", "edit", "reject"]

# What the user may not allow for a whole session: one approval per deletion/kill.
_NO_SESSION_ALLOW = {Category.DESTRUCTIVE}
_TARGET_ARGS = ("command", "file_path", "path", "pattern", "query")
_STATE_KEY = "kartrix_approvals"


@dataclass(frozen=True)
class ApprovalRequest:
    """One tool call waiting for the user (JSON-safe: it is stored in the checkpoint)."""

    tool_call_id: str
    tool: str
    command: str
    directory: str
    category: str
    reason: str
    mode: str
    allow_session: bool  # may the user allow this command for the rest of the session?
    alongside: list[str] = field(default_factory=list)  # other tool calls of the same turn
    task_key: str | None = None
    editable: bool = True  # commands can be edited; MCP tool calls can't


@dataclass(frozen=True)
class ApprovalDecision:
    type: DecisionType
    command: str | None = None  # "edit": the command to run instead
    message: str | None = None  # "reject": optional reason, passed to the agent


Approver = Callable[[list[ApprovalRequest]], Awaitable[list[ApprovalDecision]]]


# ── middleware (graph side) ───────────────────────────────────────────


class _ApprovalState(AgentState[Any]):
    # tool-call id → None (approved as is) or the user's replacement command
    kartrix_approvals: NotRequired[Annotated[dict[str, str | None], PrivateStateAttr]]


def _describe(call: ToolCall) -> str:
    args = call.get("args") or {}
    target = next((str(args[k]) for k in _TARGET_ARGS if args.get(k)), "")
    return f"{call['name']} {target[:200]}".strip()


def _mode_label() -> str:
    mode = permissions.get_mode()
    if mode != permissions.get_configured_mode():
        return f"{mode} (auto paused: external content was read in this request)"
    return mode


def _decline_message(call: ToolCall, reason: str | None) -> ToolMessage:
    text = f"Error: the user declined this {'command' if call['name'] == 'run_command' else 'tool call'}"
    text += f": {reason}" if reason else "."
    text += " It was not run. Don't run it again unless the user asks; continue without it or ask them how to proceed."
    return ToolMessage(content=text, name=call["name"], tool_call_id=call["id"] or "", status="error")


class ApprovalMiddleware(AgentMiddleware):
    """Pause the agent for the user's approval before running commands that need it."""

    state_schema = _ApprovalState

    def _external_request(self, call: ToolCall, others: list[ToolCall]) -> ApprovalRequest | None:
        ext = external_tools.get(call["name"])
        if ext is None or not ext.needs_approval or permissions.get_mode() == "read_only":
            return None  # read-only mode: the content guard refuses the call without asking
        args = json.dumps(call.get("args") or {}, ensure_ascii=False, sort_keys=True)
        return ApprovalRequest(
            tool_call_id=str(call["id"]),
            tool=call["name"],
            command=f"{ext.server}: {call['name']} {args}",
            directory="",
            category="external",
            reason=ext.reason,
            mode=permissions.get_mode(),
            allow_session=False,
            alongside=[_describe(o) for o in others if o is not call],
            task_key=audit.scope_ids().get("task_key"),
            editable=False,
        )

    def _request(self, call: ToolCall, others: list[ToolCall]) -> tuple[ApprovalRequest, Decision | None] | None:
        if not call.get("id"):
            return None
        if call["name"] != "run_command":
            external = self._external_request(call, others)
            return (external, None) if external else None
        args = call.get("args") or {}
        command, directory = args.get("command"), args.get("directory", ".")
        if not isinstance(command, str) or not isinstance(directory, str):
            return None  # the tool rejects malformed arguments itself
        try:
            decision = evaluate(command, directory, log=False)  # the tool logs its own evaluation
        except Exception:
            logger.exception("Policy evaluation failed while checking for approval")
            return None
        if not permissions.needs_approval(decision):
            return None
        request = ApprovalRequest(
            tool_call_id=str(call["id"]),
            tool=call["name"],
            command=command,
            directory=directory,
            category=str(decision.category),
            reason=decision.reason,
            mode=_mode_label(),
            allow_session=decision.category not in _NO_SESSION_ALLOW,
            alongside=[_describe(o) for o in others if o is not call],
            task_key=audit.scope_ids().get("task_key"),
        )
        return request, decision

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        calls = list(last.tool_calls) if isinstance(last, AIMessage) else []
        # Evaluate as the tools will run: auto is capped to default after external content.
        with external_content_cap(messages):
            pending = [r for c in calls if (r := self._request(c, calls)) is not None]
        if not pending:
            return {_STATE_KEY: {}} if state.get(_STATE_KEY) else None

        # On the first pass this pauses the graph; when resumed it returns the decisions.
        answer = interrupt({"type": INTERRUPT_TYPE, "requests": [asdict(r) for r, _ in pending]})
        decisions: dict[str, Any] = (answer or {}).get("decisions", {}) if isinstance(answer, dict) else {}

        approved: dict[str, str | None] = {}
        declined: list[ToolMessage] = []
        for request, decision in pending:
            call = next(c for c in calls if c["id"] == request.tool_call_id)
            raw = decisions.get(request.tool_call_id)
            kind = raw.get("type") if isinstance(raw, dict) else None
            if kind == "approve" or (kind == "approve_session" and request.allow_session and decision):
                if kind == "approve_session" and decision is not None:
                    permissions.allow_for_session(decision)
                approved[request.tool_call_id] = None
            elif (
                kind == "edit" and request.editable and isinstance(raw, dict) and str(raw.get("command") or "").strip()
            ):
                approved[request.tool_call_id] = str(raw["command"]).strip()
            elif kind == "reject":
                declined.append(_decline_message(call, raw.get("message") if isinstance(raw, dict) else None))
            else:  # no (valid) decision for this call
                declined.append(_decline_message(call, "no approval decision was received for it"))
        changes: dict[str, Any] = {_STATE_KEY: approved}
        if declined:
            # Declined calls are answered here; the tools node only runs calls without a result.
            changes["messages"] = declined
        return changes

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        approvals = (request.state or {}).get(_STATE_KEY) or {}
        call_id = request.tool_call.get("id")
        if call_id not in approvals:
            return await handler(request)
        edited = approvals[call_id]
        if edited is not None:
            # The model's message keeps its own call (provider-safe); the user's command runs.
            args = {**(request.tool_call.get("args") or {}), "command": edited}
            request = request.override(tool_call={**request.tool_call, "args": args})
        with permissions.user_approved():
            result = await handler(request)
        if edited is not None and isinstance(result, ToolMessage):
            notice = f"Note: the user replaced your command with `{edited}` before approving; this is its output.\n"
            if isinstance(result.content, str):
                result = result.model_copy(update={"content": notice + result.content})
        return result


# ── client side ───────────────────────────────────────────────────────


def _approval_interrupts(interrupts: Any) -> list[Interrupt]:
    found = list(interrupts or ())
    for intr in found:
        value = intr.value
        if not (isinstance(value, dict) and value.get("type") == INTERRUPT_TYPE):
            raise RuntimeError(f"unexpected interrupt from the agent: {value!r}")
    return found


def _session_id(config: dict[str, Any]) -> str | None:
    thread = (config.get("configurable") or {}).get("thread_id")
    return str(thread) if audit.as_uuid(thread) else audit.scope_ids().get("session_id")


async def _open_rows(requests: list[ApprovalRequest], session_id: str | None) -> list[Any]:
    """Insert one pending ``approvals`` row per request; older pending rows for the same
    tool calls (from a run that was interrupted before the user answered) expire."""
    scope = audit.scope_ids()
    try:
        async with session_scope() as s:
            await s.execute(
                update(Approval)
                .where(Approval.status == ApprovalStatus.PENDING)
                .where(Approval.request["tool_call_id"].astext.in_([r.tool_call_id for r in requests]))
                .values(status=ApprovalStatus.EXPIRED, decided_at=datetime.now(UTC))
            )
            rows = [
                Approval(
                    session_id=audit.as_uuid(session_id),
                    project_id=audit.as_uuid(scope.get("project_id")),
                    kind=ApprovalKind.COMMAND,
                    request=audit.redact_json(asdict(r)),
                )
                for r in requests
            ]
            s.add_all(rows)
            await s.flush()
            return [row.id for row in rows]
    except Exception as e:
        logger.error("Could not store approval requests", extra={"error": str(e)})
        return [None] * len(requests)


_OUTCOME = {"approve": "approved", "approve_session": "approved_session", "edit": "edited", "reject": "declined"}


async def _close_rows(
    requests: list[ApprovalRequest], decisions: list[ApprovalDecision], ids: list[Any], session_id: str | None
) -> None:
    now = datetime.now(UTC)
    for request, decision, row_id in zip(requests, decisions, ids, strict=True):
        reason = {
            "approve_session": "allowed for this session",
            "edit": f"replaced with: {decision.command}",
        }.get(decision.type, decision.message)
        status = ApprovalStatus.REJECTED if decision.type == "reject" else ApprovalStatus.APPROVED
        if row_id is not None:
            try:
                async with session_scope() as s:
                    await s.execute(
                        update(Approval)
                        .where(Approval.id == row_id)
                        .values(
                            status=status,
                            decision_reason=redact(reason) if reason else None,
                            decided_by="user",
                            decided_at=now,
                        )
                    )
            except Exception as e:
                logger.error("Could not store an approval decision", extra={"error": str(e)})
        await audit.record(
            actor="user",
            action=f"approval.{request.tool}",
            target=request.command,
            outcome=_OUTCOME[decision.type],
            session_id=session_id,
            details={
                "approval_id": str(row_id) if row_id else None,
                "tool_call_id": request.tool_call_id,
                "directory": request.directory,
                "category": request.category,
                "reason": request.reason,
                "mode": request.mode,
                "edited_command": decision.command if decision.type == "edit" else None,
                "message": decision.message,
            },
        )


async def resolve(interrupt_value: dict[str, Any], approver: Approver, session_id: str | None) -> dict[str, Any]:
    """Ask the approver about one approval interrupt; returns the resume value."""
    requests = [ApprovalRequest(**r) for r in interrupt_value.get("requests", [])]
    ids = await _open_rows(requests, session_id)
    start = time.perf_counter()
    decisions = await approver(requests)
    if len(decisions) != len(requests):
        raise RuntimeError(f"approver returned {len(decisions)} decisions for {len(requests)} requests")
    logger.info(
        "Approval decisions",
        extra={
            "decisions": [d.type for d in decisions],
            "commands": [r.command for r in requests],
            "wait_s": round(time.perf_counter() - start, 1),
        },
    )
    await _close_rows(requests, decisions, ids, session_id)
    return {"decisions": {r.tool_call_id: asdict(d) for r, d in zip(requests, decisions, strict=True)}}


async def _resume_value(interrupts: list[Interrupt], approver: Approver, config: dict[str, Any]) -> Command[Any]:
    sid = _session_id(config)
    return Command(resume={i.id: await resolve(i.value, approver, sid) for i in interrupts})


async def stream_agent(
    agent: Any, payload: Any, config: dict[str, Any], approver: Approver
) -> AsyncIterator[dict[str, Any]]:
    """``agent.astream(..., stream_mode="values")`` that answers approval interrupts and resumes.
    Yields every state; the last one is the final state."""
    next_input = payload
    while True:
        interrupts: list[Interrupt] = []
        async for chunk in agent.astream(next_input, config, stream_mode="values"):
            if "__interrupt__" in chunk:
                interrupts.extend(_approval_interrupts(chunk["__interrupt__"]))
                continue
            yield chunk
        if not interrupts:
            return
        next_input = await _resume_value(interrupts, approver, config)


async def run_agent(agent: Any, payload: Any, config: dict[str, Any], approver: Approver) -> dict[str, Any]:
    """Run the agent to completion, asking the approver whenever it pauses; the final state."""
    final: dict[str, Any] = {"messages": []}
    async for state in stream_agent(agent, payload, config, approver):
        final = state
    return final


async def resume_pending(agent: Any, config: dict[str, Any], approver: Approver) -> dict[str, Any] | None:
    """If the thread stopped at an unanswered approval (e.g. Kartrix was closed while asking),
    ask again and finish the run. None when nothing was pending."""
    state = await agent.aget_state(config)
    interrupts = _approval_interrupts(getattr(state, "interrupts", ()))
    if not interrupts:
        return None
    return await run_agent(agent, await _resume_value(interrupts, approver, config), config, approver)
