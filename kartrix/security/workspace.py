"""Workspace jail (B1): every path a tool touches must resolve inside the workspace root.

A path given by the model is resolved like this:

1. Rejected outright if empty, contains NUL, or (on Windows) uses a device/UNC prefix,
   a drive-relative form (``C:foo``), an alternate data stream (``a.txt:x``), a reserved
   device name (``nul``, ``com1.txt``) or a component ending in a dot or space (Windows
   silently strips those, so ``.env.`` would open ``.env``).
2. Relative paths are taken relative to the root; absolute paths are allowed only if
   they point inside it.
3. Symlinks and junctions are resolved (``realpath``) and the *real* target must still be
   inside the root, so a link can't be used to escape.
4. The root-relative path is checked against ``workspace.deny_read`` / ``deny_write``
   (gitignore syntax): secrets, ``.git/`` (hooks would run code), Kartrix's own state.

Residual risk: a symlink swapped in between the check and the open (TOCTOU). Writes go
through a temp file + ``os.replace`` so they never write *through* a link; the sandbox
(step 1.7) is the real boundary for anything that runs code.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pathspec import GitIgnoreSpec

from kartrix.config import settings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

Access = Literal["read", "write"]

_WINDOWS = os.name == "nt"
# NTFS and APFS are case-insensitive by default: ".ENV" is ".env".
CASE_INSENSITIVE = _WINDOWS or sys.platform == "darwin"
_RESERVED_NAME = re.compile(r"^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)(\..*)?$", re.IGNORECASE)


class WorkspaceError(Exception):
    """A path was rejected by the jail. The message is safe to show to the model."""


def check_windows_path(raw: str) -> None:
    """Reject path forms that are dangerous or ambiguous on Windows. Pure; testable anywhere."""
    if raw.startswith(("\\\\", "//")):
        raise WorkspaceError(f"UNC and device paths are not allowed: {raw}")
    if re.match(r"^[A-Za-z]:(?![\\/])", raw):
        raise WorkspaceError(f"drive-relative paths are not allowed: {raw}")
    body = raw[2:] if re.match(r"^[A-Za-z]:", raw) else raw
    for part in re.split(r"[\\/]+", body):
        if part in ("", ".", ".."):
            continue
        if ":" in part:
            raise WorkspaceError(f"alternate data streams are not allowed: {raw}")
        if part[-1] in ". ":
            raise WorkspaceError(f"path components may not end with a dot or space: {raw}")
        if _RESERVED_NAME.match(part):
            raise WorkspaceError(f"reserved device name in path: {raw}")


def _spec(patterns: list[str]) -> GitIgnoreSpec:
    return GitIgnoreSpec.from_lines([p.lower() for p in patterns] if CASE_INSENSITIVE else patterns)


@dataclass(frozen=True)
class Workspace:
    """The directory tree the agent may read and write. Build with :meth:`create`."""

    root: Path
    deny_read: GitIgnoreSpec
    deny_write: GitIgnoreSpec

    @classmethod
    def create(cls, root: str | Path) -> Workspace:
        real = Path(os.path.realpath(root))
        if not real.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {root}")
        cfg = settings.workspace
        return cls(root=real, deny_read=_spec(cfg.deny_read), deny_write=_spec(cfg.deny_write))

    def relative(self, path: Path) -> str:
        """Root-relative POSIX form of an already-resolved path ("." for the root)."""
        rel = path.relative_to(self.root).as_posix()
        return rel or "."

    def resolve(self, raw: str, access: Access = "read") -> Path:
        """Return the real absolute path for ``raw`` or raise :class:`WorkspaceError`."""
        if not raw or not raw.strip():
            raise WorkspaceError("path cannot be empty")
        if "\x00" in raw:
            raise WorkspaceError("path contains a NUL byte")
        if _WINDOWS:
            check_windows_path(raw)

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        real = Path(os.path.realpath(candidate))
        if not _is_within(real, self.root):
            logger.warning("Workspace escape blocked", extra={"path": raw, "resolved": str(real), "access": access})
            hint = " (paths are relative to the workspace root; drop the leading slash)" if raw[0] in "/\\" else ""
            raise WorkspaceError(f"access denied: {raw} is outside the workspace{hint}")

        rel = self.relative(real)
        if rel != ".":
            spec = self.deny_write if access == "write" else self.deny_read
            if _denied(spec, real, rel):
                logger.warning("Protected path blocked", extra={"path": rel, "access": access})
                raise WorkspaceError(f"access denied: {rel} is protected ({access})")
        return real

    def is_readable(self, path: Path) -> bool:
        """For walkers: True if an already-resolved path inside the root may be read."""
        rel = self.relative(path)
        return rel == "." or not _denied(self.deny_read, path, rel)

    def is_writable(self, path: Path) -> bool:
        """For walkers: True if an already-resolved path inside the root may be written."""
        rel = self.relative(path)
        return rel == "." or not _denied(self.deny_write, path, rel)


def _is_within(path: Path, root: Path) -> bool:
    p, r = os.path.normcase(str(path)), os.path.normcase(str(root))
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _denied(spec: GitIgnoreSpec, path: Path, rel: str) -> bool:
    key = rel.lower() if CASE_INSENSITIVE else rel
    # Check every ancestor as a directory too, so "secrets/" also covers "secrets/a/b".
    parts = key.split("/")
    for i in range(1, len(parts)):
        if spec.match_file("/".join(parts[:i]) + "/"):
            return True
    return spec.match_file(key + "/" if path.is_dir() else key)


_current: Workspace | None = None


def set_workspace(root: str | Path) -> Workspace:
    """Jail all file tools to ``root`` (the repo Kartrix was started in)."""
    global _current
    _current = Workspace.create(root)
    logger.info("Workspace set", extra={"root": str(_current.root)})
    return _current


def get_workspace() -> Workspace:
    """The active workspace; defaults to the current directory if none was set."""
    return _current if _current is not None else set_workspace(Path.cwd())
