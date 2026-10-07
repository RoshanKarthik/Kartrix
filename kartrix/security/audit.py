"""Audit log (B10) and the tool-call choke point.

:class:`AuditMiddleware` wraps every tool call of every agent (native and MCP tools):

- the tool's result is **redacted** (``kartrix.security.secrets``) before the model sees it;
- one row is appended to ``audit_log`` (append-only table): tool, redacted + clipped
  arguments, outcome, duration, output size, how many secrets were redacted, plus notes
  the tool added via :func:`note` (e.g. the command-policy decision).

If the database write fails the record goes to the log file at ERROR level instead, and
the tool call still succeeds — losing the agent's work over an audit hiccup would be worse.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from sqlalchemy import select

from kartrix.core.events import ToolCallFinished, ToolCallStarted, emit
from kartrix.db.engine import session_scope
from kartrix.db.models import AuditLog
from kartrix.observability.logger import get_logger
from kartrix.security.secrets import redact, redact_with_count

logger = get_logger(__name__)

_ARG_CHARS = 2000  # per string argument
_PREVIEW_CHARS = 500
_TARGET_ARGS = ("command", "file_path", "path", "directory", "pattern", "query", "skill_name", "name")

_scope: ContextVar[dict[str, Any] | None] = ContextVar("kartrix_audit_scope", default=None)
_notes: ContextVar[dict[str, Any] | None] = ContextVar("kartrix_audit_notes", default=None)


@contextmanager
def audit_scope(**ids: Any) -> Iterator[None]:
    """Attach ids (session_id, project_id, task_key, …) to audit rows written inside."""
    token = _scope.set({**(_scope.get() or {}), **ids})
    try:
        yield
    finally:
        _scope.reset(token)


def scope_ids() -> dict[str, Any]:
    """The ids set by the enclosing :func:`audit_scope` calls."""
    return dict(_scope.get() or {})


def as_uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except ValueError:
        return None


def note(**fields: Any) -> None:
    """Add details to the audit row of the tool call currently running (no-op outside one).

    Works from sync tools running in a worker thread too: the thread gets a copy of the
    context, which still points at the same notes dict."""
    holder = _notes.get()
    if holder is not None:
        holder.update(fields)


def redact_json(value: Any) -> Any:
    return json.loads(redact(json.dumps(value, default=str, ensure_ascii=False)))


async def record(
    *,
    action: str,
    outcome: str,
    actor: str = "agent",
    target: str | None = None,
    details: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    """Append one audit row (secrets redacted). Never raises."""
    scope = _scope.get() or {}
    extra = {k: v for k, v in scope.items() if k not in ("session_id", "project_id", "task_id")}
    row: dict[str, Any] = {
        "session_id": as_uuid(session_id) or as_uuid(scope.get("session_id")),
        "project_id": as_uuid(scope.get("project_id")),
        "task_id": as_uuid(scope.get("task_id")),
        "actor": actor,
        "action": action[:100],
        "target": redact(target)[:1000] if target else None,
        "outcome": outcome[:32],
        "details": redact_json({**extra, **(details or {})}),
    }
    try:
        async with session_scope() as s:
            s.add(AuditLog(**row))
    except Exception as e:
        logger.error("Audit write failed; record kept here instead", extra={"audit": row, "error": str(e)})


async def recent(session_id: str, limit: int = 20) -> list[AuditLog]:
    """The newest audit rows of a session, newest first."""
    sid = as_uuid(session_id)
    async with session_scope() as s:
        result = await s.execute(
            select(AuditLog).where(AuditLog.session_id == sid).order_by(AuditLog.id.desc()).limit(limit)
        )
        return list(result.scalars())


# ── middleware ────────────────────────────────────────────────────────


def _clip_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > _ARG_CHARS:
            value = f"{value[:_ARG_CHARS]}… [{len(value) - _ARG_CHARS} more chars]"
        out[key] = value
    return out


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content if isinstance(b, str | dict)
        )
    return str(content)


def _redact_content(content: Any) -> tuple[Any, int]:
    if isinstance(content, str):
        return redact_with_count(content)
    if isinstance(content, list):
        total = 0
        blocks: list[Any] = []
        for b in content:
            if isinstance(b, str):
                text, n = redact_with_count(b)
                blocks.append(text)
            elif isinstance(b, dict) and isinstance(b.get("text"), str):
                text, n = redact_with_count(b["text"])
                blocks.append({**b, "text": text})
            else:
                n = 0
                blocks.append(b)
            total += n
        return blocks, total
    return content, 0


def classify_outcome(text: str, status: str | None = None) -> str:
    head = text[:300]
    if head.startswith(("Error: command denied", "Error: access denied")) or "is protected" in head:
        return "denied"
    if head.startswith("Error: command not run"):
        return "needs_approval"
    if head.startswith("Error: the user declined"):
        return "declined"
    if head.startswith("Error: stopped"):
        return "stopped"
    if head.startswith("Error") or status == "error":
        return "error"
    return "ok"


class AuditMiddleware(AgentMiddleware):
    """Redact every tool result and write an audit row for every tool call."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        holder: dict[str, Any] = {}
        token = _notes.set(holder)
        start = time.perf_counter()
        call = request.tool_call
        args = call.get("args") or {}
        emit(
            ToolCallStarted(
                call_id=call.get("id"),
                tool=call.get("name", "?"),
                target=next((str(args[k])[:300] for k in _TARGET_ARGS if args.get(k)), None),
                task_key=scope_ids().get("task_key"),
            )
        )
        try:
            result = await handler(request)
        except asyncio.CancelledError:  # the kill switch cancelled the run mid-call
            await self._record(request, "stopped", holder, start, {"exception": "cancelled"})
            raise
        except Exception as e:
            await self._record(request, "error", holder, start, {"exception": f"{type(e).__name__}: {e}"})
            raise
        finally:
            _notes.reset(token)

        if not isinstance(result, ToolMessage):
            await self._record(request, "ok", holder, start, {"result": "command"})
            return result
        content, secrets = _redact_content(result.content)
        if secrets:
            result = result.model_copy(update={"content": content})
        text = _content_text(content)
        details = {"output_chars": len(text), "output_preview": text[:_PREVIEW_CHARS], "secrets_redacted": secrets}
        await self._record(request, classify_outcome(text, result.status), holder, start, details)
        return result

    async def _record(
        self, request: ToolCallRequest, outcome: str, notes: dict[str, Any], start: float, details: dict[str, Any]
    ) -> None:
        call = request.tool_call
        args = call.get("args") or {}
        target = next((str(args[k]) for k in _TARGET_ARGS if args.get(k)), None)
        config = getattr(request.runtime, "config", None) or {}
        session_id = (config.get("configurable") or {}).get("thread_id")
        final = notes.get("outcome", outcome)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        emit(
            ToolCallFinished(
                call_id=call.get("id"), tool=call.get("name", "?"), outcome=final, duration_ms=duration_ms,
                task_key=scope_ids().get("task_key"),
            )
        )  # fmt: skip
        await record(
            action=f"tool.{call.get('name', '?')}",
            outcome=notes.pop("outcome", outcome),
            target=target,
            session_id=session_id,
            details={
                "args": _clip_args(args),
                "tool_call_id": call.get("id"),
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                **details,
                **notes,
            },
        )
