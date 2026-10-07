"""Workspace checkpoints and ``/undo`` (B11).

Before every chat turn and every ``/plan`` task the workspace is snapshotted into a **separate
git repository** in the per-user data folder (``checkpoints/<workspace id>/repo``, used through
``GIT_DIR`` + ``GIT_WORK_TREE``). The user's own ``.git`` — branches, index, stash — is never
touched, and folders that aren't git repositories yet work too. After the turn/task a second
snapshot is taken; if anything changed, the pair becomes an undo entry.

- A snapshot is ``git add -A`` into the shadow index + ``git write-tree``. Files ignored by the
  workspace's ``.gitignore`` files, by ``checkpoints.exclude`` or by the secret patterns of
  ``workspace.deny_write`` are not snapshotted, so ``/undo`` never touches them (a negation in the
  workspace's own ``.gitignore`` can re-include a file; it then only lands in the user's own folder).
- ``/undo`` restores only the files the entry changed, to their content before it; files it
  created are deleted. If one of them changed again since (edited by hand, or by a later turn
  that was not undone), nothing happens until the user confirms (``force``). ``/redo``
  re-applies an undone entry the same way. The state just before an undo is kept as well.
- Bytes are kept exactly and git runs nothing: the global/system git config is ignored, the
  shadow repo's ``info/attributes`` switches off line-ending conversion, filters and encodings
  that a workspace's ``.gitattributes`` could ask for (a cloned repo can't make git run a filter
  program), and its hooks folder is empty.
- Trees of the newest ``checkpoints.keep`` entries are kept under ``refs/kartrix/``.

Never blocks the agent: if git is missing or a snapshot fails, the run goes on without one and the
user is told. Two Kartrix processes on the same workspace are serialised by git's index lock
(a stale lock older than a minute is removed); full multi-process safety comes with J9.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kartrix.config import settings
from kartrix.core.events import FilesChanged, emit
from kartrix.observability.logger import get_logger
from kartrix.paths import user_data_dir
from kartrix.security.command_policy import find_executable
from kartrix.security.environment import scrubbed_env
from kartrix.security.workspace import get_workspace

logger = get_logger(__name__)

_GIT_TIMEOUT = 300
_STALE_LOCK_SECONDS = 60
_MAX_LISTED_FILES = 500
_GITLINK = "160000"  # a nested repository; never restored or deleted
# Highest-precedence attributes: no eol conversion, filters (they run programs), ident or re-encoding.
_ATTRIBUTES = "* -text -eol -filter -ident -working-tree-encoding\n"


class CheckpointError(Exception):
    """A git operation of the checkpoint store failed."""


@dataclass
class Entry:
    """One undoable change: the workspace's snapshot trees before and after a turn/task."""

    id: str
    label: str
    before: str
    after: str
    created: float
    session_id: str | None = None
    files: list[str] = field(default_factory=list)  # changed paths (capped)
    file_count: int = 0


@dataclass
class UndoResult:
    entry: Entry
    applied: bool
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # changed since; blocks unless forced
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Change:
    status: str  # A (only in the target), D (not in the target), M/T (different)
    path: str
    gitlink: bool


class Checkpoints:
    """The checkpoint store of one workspace."""

    def __init__(self, root: Path, git: Path, store: Path | None = None) -> None:
        self.root = root.resolve()
        key = hashlib.sha256(os.path.normcase(str(self.root)).encode()).hexdigest()[:16]
        self.store = store or user_data_dir() / "checkpoints" / key
        self._git_dir = self.store / "repo"
        self._stack_file = self.store / "stack.json"
        self._git_exe = git
        self._lock = threading.RLock()
        self._env = self._make_env()
        self._init()

    # ── git plumbing ─────────────────────────────────────────────────

    def _make_env(self) -> dict[str, str]:
        env = {k: v for k, v in scrubbed_env().items() if not k.upper().startswith("GIT_")}
        empty_config = self.store / "empty.gitconfig"
        env.update(
            GIT_DIR=str(self._git_dir),
            GIT_WORK_TREE=str(self.root),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=str(empty_config),  # the user's global config (filters, hooks, lfs) doesn't apply
            GIT_TERMINAL_PROMPT="0",
            GIT_LITERAL_PATHSPECS="1",
            GIT_ADVICE="0",
        )
        return env

    def _git(self, *args: str, stdin: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        proc = subprocess.run(  # noqa: S603 — git resolved from PATH (never the workspace), fixed arguments
            [str(self._git_exe), *args],
            cwd=self.root,
            env=self._env,
            input=stdin,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
        if check and proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace").strip()
            raise CheckpointError(f"git {args[0]} failed: {err[:500]}")
        return proc

    def _init(self) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / "empty.gitconfig").touch()
        (self.store / "workspace.txt").write_text(str(self.root), encoding="utf-8")  # for humans
        if not (self._git_dir / "HEAD").exists():
            self._git("init", "--quiet")
            for key, value in (
                ("core.autocrlf", "false"),
                ("core.safecrlf", "false"),
                ("core.longpaths", "true"),
                ("core.fsmonitor", "false"),
                ("core.hooksPath", str(self.store / "no-hooks")),
                ("gc.autoDetach", "false"),
                ("advice.addEmbeddedRepo", "false"),
            ):
                self._git("config", key, value)
        info = self._git_dir / "info"
        info.mkdir(exist_ok=True)
        (info / "attributes").write_text(_ATTRIBUTES, encoding="utf-8")
        excludes = [".kartrix/", *settings.checkpoints.exclude, *settings.workspace.deny_write]
        (info / "exclude").write_text("\n".join(excludes) + "\n", encoding="utf-8")

    def _remove_stale_lock(self) -> bool:
        lock = self._git_dir / "index.lock"
        try:
            if time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS:
                lock.unlink()
                logger.warning("Removed a stale checkpoint index lock", extra={"lock": str(lock)})
                return True
        except OSError:
            pass
        return False

    def snapshot(self) -> str:
        """Snapshot the workspace; returns the tree id."""
        with self._lock:
            for attempt in range(2):
                # --ignore-errors: a file another program holds open (Windows) is skipped, not fatal.
                proc = self._git("add", "-A", "--ignore-errors", "--", ".", check=False)
                if proc.returncode == 0:
                    break
                err = proc.stderr.decode("utf-8", "replace")
                if "index.lock" in err and attempt == 0 and self._remove_stale_lock():
                    continue
                if "index.lock" in err:
                    raise CheckpointError(f"git add failed: {err.strip()[:500]}")
                logger.warning("Some files could not be snapshotted", extra={"stderr": err.strip()[:1000]})
                break
            return self._git("write-tree").stdout.decode().strip()

    def _changes(self, src: str, dst: str) -> list[_Change]:
        """What differs from tree ``src`` to tree ``dst``."""
        out = self._git("diff", "--raw", "-z", "--no-renames", "--no-abbrev", src, dst).stdout
        parts = out.decode("utf-8", "surrogateescape").split("\0")
        changes: list[_Change] = []
        for meta, path in zip(parts[0::2], parts[1::2], strict=False):
            if not meta.startswith(":"):
                continue
            old_mode, new_mode, *_, status = meta[1:].split()
            changes.append(_Change(status[0], path, _GITLINK in (old_mode, new_mode)))
        return changes

    def _ref(self, entry_id: str, name: str, tree: str | None) -> None:
        ref = f"refs/kartrix/{entry_id}/{name}"
        if tree is None:
            self._git("update-ref", "-d", ref, check=False)
        else:
            self._git("update-ref", ref, tree)

    # ── undo stack ───────────────────────────────────────────────────

    def _load(self) -> dict[str, list[Entry]]:
        try:
            data = json.loads(self._stack_file.read_text(encoding="utf-8"))
            return {k: [Entry(**e) for e in data.get(k, [])] for k in ("undo", "redo")}
        except (OSError, ValueError, TypeError):
            return {"undo": [], "redo": []}

    def _save(self, stack: dict[str, list[Entry]]) -> None:
        tmp = self._stack_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: [asdict(e) for e in v] for k, v in stack.items()}, indent=1), encoding="utf-8")
        tmp.replace(self._stack_file)

    def _drop(self, entry: Entry) -> None:
        for name in ("before", "after", "overwritten"):
            self._ref(entry.id, name, None)

    def entries(self) -> tuple[list[Entry], list[Entry]]:
        """(undo entries, redo entries), oldest first."""
        with self._lock:
            stack = self._load()
            return stack["undo"], stack["redo"]

    def record(self, label: str, before: str, after: str, session_id: str | None = None) -> Entry | None:
        """Add an undo entry if the trees differ (a new change clears the redo list)."""
        if before == after:
            return None
        with self._lock:
            paths = [c.path for c in self._changes(before, after)]
            entry = Entry(
                id=uuid.uuid4().hex[:12],
                label=label[:200],
                before=before,
                after=after,
                created=time.time(),
                session_id=session_id,
                files=paths[:_MAX_LISTED_FILES],
                file_count=len(paths),
            )
            self._ref(entry.id, "before", before)
            self._ref(entry.id, "after", after)
            stack = self._load()
            for old in stack["redo"]:
                self._drop(old)
            stack["redo"] = []
            stack["undo"].append(entry)
            keep = settings.checkpoints.keep
            for old in stack["undo"][:-keep]:
                self._drop(old)
            stack["undo"] = stack["undo"][-keep:]
            self._save(stack)
            return entry

    def undo(self, force: bool = False) -> UndoResult | None:
        """Undo the newest entry; None if there is nothing to undo."""
        return self._step("undo", "redo", force)

    def redo(self, force: bool = False) -> UndoResult | None:
        """Re-apply the newest undone entry; None if there is nothing to redo."""
        return self._step("redo", "undo", force)

    def _step(self, source: str, target: str, force: bool) -> UndoResult | None:
        with self._lock:
            stack = self._load()
            if not stack[source]:
                return None
            entry = stack[source][-1]
            src, dst = (entry.after, entry.before) if source == "undo" else (entry.before, entry.after)
            result = self._move(entry, src, dst, force)
            if result.applied:
                stack[source].pop()
                stack[target].append(entry)
                self._save(stack)
            return result

    def _move(self, entry: Entry, src: str, dst: str, force: bool) -> UndoResult:
        """Make the files that differ between ``src`` and ``dst`` look like ``dst`` again,
        provided they still look like ``src`` (else report conflicts, unless forced)."""
        now = self.snapshot()
        changes = [c for c in self._changes(src, dst) if not c.gitlink]
        touched = {c.path for c in changes}
        conflicts = sorted(c.path for c in self._changes(src, now) if c.path in touched) if now != src else []
        if conflicts and not force:
            return UndoResult(entry, applied=False, conflicts=conflicts)
        if conflicts:
            self._ref(entry.id, "overwritten", now)  # the state the user overrode stays recoverable

        restore = sorted(c.path for c in changes if c.status != "D")
        delete = sorted(c.path for c in changes if c.status == "D")
        result = UndoResult(entry, applied=True, conflicts=conflicts)
        if restore:
            pathspec = "\0".join(restore).encode("utf-8", "surrogateescape")
            self._git("checkout", dst, "--pathspec-from-file=-", "--pathspec-file-nul", stdin=pathspec)
            result.restored = restore
        for rel in delete:
            try:
                if self._delete(rel):
                    result.deleted.append(rel)
            except OSError as e:
                result.errors.append(f"{rel}: {e.strerror or e}")
        self.snapshot()  # keep the shadow index in step with the workspace
        logger.info(
            "Checkpoint applied",
            extra={"entry": entry.id, "restored": len(result.restored), "deleted": len(result.deleted)},
        )
        return result

    def _delete(self, rel: str) -> bool:
        """Delete a file the entry created — only inside the workspace, never through a link."""
        parts = Path(rel).parts
        if not parts or Path(rel).is_absolute() or ".." in parts:
            raise OSError(f"refusing unexpected path {rel!r}")
        path = self.root / rel
        parent = Path(os.path.realpath(path.parent))
        if parent != self.root and self.root not in parent.parents:
            raise OSError(f"{rel} is outside the workspace")
        if not path.is_symlink() and not path.is_file():
            return False  # already gone, or a directory (never deleted)
        path.unlink()
        # Remove folders that became empty (they existed only for the deleted files).
        for folder in path.parents:
            if folder == self.root or self.root not in folder.parents:
                break
            try:
                folder.rmdir()
            except OSError:
                break
        return True


# ── the current workspace ─────────────────────────────────────────────

_store: Checkpoints | None = None
_unavailable: str | None = None


def setup() -> str | None:
    """Open the checkpoint store of the current workspace. Returns why checkpoints are off, or None."""
    global _store, _unavailable
    _store, _unavailable = None, None
    if not settings.checkpoints.enabled:
        _unavailable = "turned off (checkpoints.enabled: false)"
        return _unavailable
    root = get_workspace().root
    git = find_executable("git", os.environ.get("PATH", ""), root)
    if git is None:
        _unavailable = "git was not found on PATH — install git to enable checkpoints and /undo"
        return _unavailable
    try:
        _store = Checkpoints(root, git)
    except (OSError, CheckpointError, subprocess.SubprocessError) as e:
        _unavailable = f"could not open the checkpoint store: {e}"
        logger.error("Checkpoints unavailable", extra={"error": str(e)})
    return _unavailable


def current() -> Checkpoints | None:
    return _store


def unavailable_reason() -> str | None:
    return _unavailable


@dataclass
class Tracked:
    """Filled in when a :func:`track` block ends: the undo entry, if files changed."""

    entry: Entry | None = None


@asynccontextmanager
async def track(label: str, session_id: str | None = None) -> AsyncIterator[Tracked]:
    """Snapshot before and after the block; an undo entry is kept if files changed (and a
    :class:`~kartrix.core.events.FilesChanged` event emitted). Errors never stop the block —
    the change just isn't undoable."""
    store = _store
    tracked = Tracked()
    before: str | None = None
    if store is not None:
        try:
            before = await asyncio.to_thread(store.snapshot)
        except (OSError, CheckpointError, subprocess.SubprocessError) as e:
            logger.error("Checkpoint before a run failed", extra={"error": str(e), "label": label})
    try:
        yield tracked
    finally:
        if store is not None and before is not None:
            try:
                after = await asyncio.to_thread(store.snapshot)
                tracked.entry = await asyncio.to_thread(store.record, label, before, after, session_id)
            except (OSError, CheckpointError, subprocess.SubprocessError) as e:
                logger.error("Checkpoint after a run failed", extra={"error": str(e), "label": label})
            if tracked.entry is not None:
                entry = tracked.entry
                emit(FilesChanged(label=entry.label, files=list(entry.files), count=entry.file_count))
