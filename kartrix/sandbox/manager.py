"""Which sandbox runs commands here, and what each command gets.

``sandbox.backend``:

- ``auto`` — the OS-native sandbox: macOS Seatbelt; Linux bubblewrap, else Landlock; Windows
  AppContainer. None available → no sandbox (commands that run code need approval).
- ``native`` — the same, but a missing sandbox is an error at the first command.
- ``docker`` — Docker (``sandbox.docker.image``); unavailable → no sandbox, with the reason.
- ``none`` — no sandbox.

Detection runs once per process. ``git`` never runs sandboxed: it is a trusted program that
must write ``.git`` (protected inside the sandbox) and use the user's credentials; the command
policy already decides what git may do.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.sandbox.base import Backend, Network, SandboxUnavailable
from kartrix.security.command_rules import Category, program_name

logger = get_logger(__name__)

UNSANDBOXED_PROGRAMS = {"git"}


@dataclass
class SandboxStatus:
    backend: Backend | None
    tried: list[str] = field(default_factory=list)  # "<name>: why it isn't available"

    def describe(self) -> str:
        if self.backend is not None:
            return self.backend.describe()
        if settings.sandbox.backend == "none":
            return "off (sandbox.backend: none) — every command that runs code needs your approval"
        why = "; ".join(self.tried) or "not supported on this OS"
        return f"not available ({why}) — every command that runs code needs your approval"


def _native_candidates() -> list[type[Backend]]:
    platform = sys.platform  # a variable: type-checks every branch on every OS
    if platform == "win32":
        from kartrix.sandbox.windows import AppContainerBackend

        return [AppContainerBackend]
    from kartrix.sandbox.posix import BubblewrapBackend, LandlockBackend, SeatbeltBackend

    if platform == "darwin":
        return [SeatbeltBackend]
    if platform.startswith("linux"):
        return [BubblewrapBackend, LandlockBackend]
    return []


def detect() -> SandboxStatus:
    choice = settings.sandbox.backend
    if choice == "none":
        return SandboxStatus(None)
    if choice == "docker":
        from kartrix.sandbox.docker import DockerBackend

        candidates: list[type[Backend]] = [DockerBackend]
    else:
        candidates = _native_candidates()
    status = SandboxStatus(None)
    for cls in candidates:
        try:
            status.backend = cls.detect()  # type: ignore[attr-defined]
            break
        except SandboxUnavailable as e:
            status.tried.append(f"{cls.name}: {e}")
    logger.info("Sandbox detected", extra={"backend": status.backend and status.backend.name, "tried": status.tried})
    return status


_status: SandboxStatus | None = None
_lock = threading.Lock()


def status() -> SandboxStatus:
    global _status
    with _lock:
        if _status is None:
            _status = detect()
        return _status


def get_sandbox() -> Backend | None:
    return status().backend


def set_sandbox(backend: Backend | None) -> None:
    """Use ``backend`` (tests, or after a config change); ``reset_sandbox`` detects again."""
    global _status
    with _lock:
        _status = SandboxStatus(backend)


def reset_sandbox() -> None:
    global _status
    with _lock:
        _status = None


def require_native() -> bool:
    return settings.sandbox.backend == "native"


def argv_unsandboxed(argv: list[str]) -> bool:
    """True for programs that always run outside the sandbox (git)."""
    return bool(argv) and program_name(argv[0]) in UNSANDBOXED_PROGRAMS


def sandbox_for(argv: list[str], exe: Path | None = None) -> Backend | None:
    """The backend that runs ``argv`` (``exe``: its resolved executable), or None if it runs unsandboxed."""
    if argv_unsandboxed(argv):
        return None
    backend = get_sandbox()
    if backend is not None and backend.cannot_run(argv, exe):
        return None
    return backend


def unsandboxed_reason(argv: list[str], exe: Path | None = None) -> str | None:
    """Why the active sandbox can't run ``argv`` (None: it can, or there is no sandbox)."""
    backend = get_sandbox()
    return backend.cannot_run(argv, exe) if backend is not None and not argv_unsandboxed(argv) else None


def network_for(category: Category, backend: Backend) -> Network:
    """Network for a command of ``category``: approved network commands get any host; installs
    get the registries (or any host where they can't be enforced — those installs need approval)."""
    if category is Category.NETWORK:
        return "full"
    if category is Category.INSTALL:
        return "registries" if backend.registries_enforced else "full"
    return "off"
