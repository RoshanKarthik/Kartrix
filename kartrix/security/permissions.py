"""Permission modes (B4) and the approval hook used by "ask" decisions.

- ``read_only`` — inspect only: read-only commands; file changes and everything else are refused.
- ``default``   — read-only commands and file edits run; anything that runs code, installs,
                  changes git, uses the network, deletes, or is unknown needs approval.
- ``auto``      — also runs tests/builds/scripts, project-local installs from allow-listed
                  registries and recoverable git changes; network, destructive and unknown
                  commands still need approval.

The hard deny list applies in every mode. Until approvals exist (step 1.4) there is no
approval handler, so "ask" means "not run".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast, get_args

from kartrix.config import settings
from kartrix.observability.logger import get_logger

if TYPE_CHECKING:
    from kartrix.security.command_policy import Decision

logger = get_logger(__name__)

Mode = Literal["read_only", "default", "auto"]
MODES: tuple[str, ...] = get_args(Mode)

ApprovalHandler = Callable[["Decision"], bool]

_mode: Mode | None = None
_approval_handler: ApprovalHandler | None = None


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


def set_approval_handler(handler: ApprovalHandler | None) -> None:
    """Install the function that asks the user about an "ask" decision (step 1.4)."""
    global _approval_handler
    _approval_handler = handler


def has_approval_handler() -> bool:
    return _approval_handler is not None


def request_approval(decision: Decision) -> bool:
    """True if the user approved; False if declined or nobody can be asked."""
    return _approval_handler(decision) if _approval_handler is not None else False
