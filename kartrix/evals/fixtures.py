"""Fixture repositories and task workspaces on disk.

- RAG fixtures (``evals/repos.yaml``) are materialised once per pinned commit under the eval cache
  (``.kartrix/evals/repos/<name>@<commit>``): this repository via ``git archive`` (so the labels keep
  matching while the code moves on), open-source repositories by a shallow fetch of exactly that
  commit. Their ``.git`` is dropped — they are data, never run.
- Agent tasks run in a fresh copy of their app (``.kartrix/evals/work/<run>/<task>-<n>``) with its own
  git repository and one initial commit, so the run's changes are a plain ``git diff``.
"""

from __future__ import annotations

import io
import os
import shutil
import stat
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kartrix.config import PROJECT_ROOT
from kartrix.evals.datasets import FixtureRepo
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_MARKER = ".kartrix-fixture"
_GIT_ID = ["-c", "user.name=Kartrix Evals", "-c", "user.email=evals@kartrix.invalid", "-c", "commit.gpgsign=false"]


class FixtureError(Exception):
    """A fixture could not be prepared (network, unknown commit, …). Safe to show."""


def cache_dir() -> Path:
    """Project-local eval cache (``KARTRIX_EVALS_CACHE`` overrides)."""
    path = Path(os.environ.get("KARTRIX_EVALS_CACHE") or PROJECT_ROOT / ".kartrix" / "evals")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _git(args: list[str], cwd: Path | None = None, timeout: float = 300) -> subprocess.CompletedProcess[bytes]:
    git = shutil.which("git")
    if git is None:
        raise FixtureError("git is not installed")
    proc = subprocess.run([git, *args], cwd=cwd, capture_output=True, timeout=timeout, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise FixtureError(f"git {' '.join(args[:3])} failed: {proc.stderr.decode(errors='replace').strip()[:500]}")
    return proc


def _on_rm_error(func: Callable[..., Any], path: str, _exc: BaseException) -> None:
    os.chmod(path, stat.S_IWRITE)  # git marks its objects read-only on Windows
    func(path)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onexc=_on_rm_error)


def ensure_repo(repo: FixtureRepo, refresh: bool = False) -> Path:
    """The fixture's files at its pinned commit (materialised on first use)."""
    target = cache_dir() / "repos" / repo.key
    marker = target / _MARKER
    if marker.is_file() and marker.read_text().strip() == repo.commit and not refresh:
        return target
    remove_tree(target)  # a half-finished earlier attempt
    target.mkdir(parents=True)
    try:
        if repo.source == "self":
            data = _git(["archive", "--format=tar", repo.commit], cwd=PROJECT_ROOT).stdout
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                tar.extractall(target, filter="data")
        else:
            assert repo.url  # noqa: S101 — validated by FixtureRepo
            _git(["init", "-q", str(target)])
            _git(["fetch", "-q", "--depth", "1", repo.url, repo.commit], cwd=target, timeout=600)
            _git(["-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"], cwd=target)
            remove_tree(target / ".git")
    except BaseException:
        remove_tree(target)
        raise
    marker.write_text(repo.commit + "\n")
    logger.info("Fixture repo ready", extra={"repo": repo.name, "commit": repo.commit, "path": str(target)})
    return target


def prepare_workspace(app_dir: Path, target: Path, overlay: Path | None = None) -> Path:
    """Copy ``app_dir`` (plus the task's ``overlay`` files) to ``target`` and commit it as the starting
    point of a task run."""
    remove_tree(target)
    shutil.copytree(app_dir, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "node_modules"))
    if overlay is not None and overlay.is_dir():
        copy_into(overlay, target)
    _git(["init", "-q", "-b", "main"], cwd=target)
    # Kartrix's own state in the workspace (sessions, checkpoints) is not part of the agent's changes.
    (target / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (target / ".git" / "info" / "exclude").write_text(".kartrix/\n__pycache__/\n.pytest_cache/\n")
    _git(["add", "-A"], cwd=target)
    _git([*_GIT_ID, "commit", "-q", "-m", "fixture"], cwd=target)
    return target


def copy_into(src: Path, workspace: Path) -> list[str]:
    """Copy every file under ``src`` into ``workspace`` (overwriting); returns the relative paths."""
    copied = []
    for path in sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = path.relative_to(src)
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append(rel.as_posix())
    return copied


def changed_files(workspace: Path) -> list[str]:
    """Files added, changed or deleted since the fixture commit (untracked files included)."""
    _git(["add", "-A"], cwd=workspace)
    out = _git(["diff", "--cached", "--name-only", "HEAD"], cwd=workspace).stdout.decode()
    _git(["reset", "-q"], cwd=workspace)
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def diff_stat(workspace: Path) -> dict[str, int]:
    """Lines added/removed since the fixture commit."""
    _git(["add", "-A"], cwd=workspace)
    out = _git(["diff", "--cached", "--numstat", "HEAD"], cwd=workspace).stdout.decode()
    _git(["reset", "-q"], cwd=workspace)
    added = removed = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            added += int(parts[0])
            removed += int(parts[1])
    return {"added": added, "removed": removed}
