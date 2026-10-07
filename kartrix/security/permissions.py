"""Permission modes (B4) and the approval hook used by "ask" decisions.

- ``read_only`` — inspect only: read-only commands; file changes and everything else are refused.
- ``default``   — read-only commands and file edits run; anything that runs code, installs,
                  changes git, uses the network, deletes, or is unknown needs approval.
- ``auto``      — also runs tests/builds/scripts, project-local installs from allow-listed
                  registries and recoverable git changes; network, destructive and unknown
                  commands still need approval.

The hard deny list applies in every mode. An "ask" decision runs only when the user approved
that exact tool call (``kartrix.security.approvals`` asks before the tool runs and marks the
call with :func:`user_approved`) or allowed the same command for the rest of the session.
Agents without the approval middleware (or no one to ask) treat "ask" as "not run".
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal, cast, get_args

from kartrix.config import settings
from kartrix.observability.logger import get_logger

if TYPE_CHECKING:
    from kartrix.security.command_policy import Decision

logger = get_logger(__name__)

Mode = Literal["read_only", "default", "auto"]
MODES: tuple[str, ...] = get_args(Mode)

_mode: Mode | None = None
_user_approved: ContextVar[bool] = ContextVar("kartrix_user_approved", default=False)
_session_allowed: set[tuple[str, str]] = set()  # (command line, directory) the user allowed for this session


class PermissionDeniedError(Exception):
    """The current permission mode forbids the action. Safe to show to the model."""


def get_mode() -> Mode:
    return _mode if _mode is not None else settings.permissions.mode


def set_mode(mode: str) -> Mode:
    if mode not in MODES:
        raise ValueError(f"unknown permission mode {mode!r} (choose from {', '.join(MODES)})")
    global _mode
    _mode = cast(Mode, mode)
    logger.info("Permission mode changed", extra={"mode": mode})
    return _mode


def ensure_writes_allowed() -> None:
    """File tools call this before changing anything."""
    if get_mode() == "read_only":
        raise PermissionDeniedError("read-only mode: file changes are disabled (the user can switch with /mode)")


@contextmanager
def user_approved() -> Iterator[None]:
    """Mark the tool call running inside as approved by the user (set by the approval middleware).

    Sync tools run in a worker thread with a copy of the context, so they see the mark too."""
    token = _user_approved.set(True)
    try:
        yield
    finally:
        _user_approved.reset(token)


def approval_key(decision: Decision) -> tuple[str, str]:
    """What "allow for this session" remembers: the exact command line and directory."""
    return shlex.join(decision.argv), str(decision.cwd)


def allow_for_session(decision: Decision) -> None:
    _session_allowed.add(approval_key(decision))
    logger.info("Command allowed for the session", extra={"command": decision.argv, "cwd": str(decision.cwd)})


def clear_session_allowances() -> None:
    """Forget "allow for this session" choices (new or switched session)."""
    _session_allowed.clear()


def needs_approval(decision: Decision) -> bool:
    """An "ask" decision the user hasn't already allowed for this session."""
    return decision.action == "ask" and approval_key(decision) not in _session_allowed


def request_approval(decision: Decision) -> bool:
    """True if the user approved this call (or this command for the session)."""
    return _user_approved.get() or not needs_approval(decision)
