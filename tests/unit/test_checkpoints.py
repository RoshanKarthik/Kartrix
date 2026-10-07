"""Workspace checkpoints and /undo (B11) — a real git, in temporary folders."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kartrix.config import settings
from kartrix.security import checkpoints
from kartrix.security.checkpoints import Checkpoints
from kartrix.security.workspace import set_workspace

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    root = tmp_path / "ws"
    root.mkdir()
    set_workspace(root)
    return root


def _store(ws: Path) -> Checkpoints:
    assert GIT
    return Checkpoints(ws, Path(GIT))


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _turn(cp: Checkpoints, change, label: str = "turn"):
    before = cp.snapshot()
    change()
    return cp.record(label, before, cp.snapshot())


def _user_git(ws: Path, *args: str) -> str:
    assert GIT
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run([GIT, *args], cwd=ws, env=env, capture_output=True, text=True, check=True)
    return proc.stdout


def test_undo_restores_modified_deleted_and_removes_created_files(ws: Path) -> None:
    _write(ws / "app.py", "v1\n")
    _write(ws / "old.txt", "keep me\n")
    cp = _store(ws)

    def change() -> None:
        _write(ws / "app.py", "v2\n")
        (ws / "old.txt").unlink()
        _write(ws / "src" / "deep" / "new.py", "new\n")

    entry = _turn(cp, change, "add feature")
    assert entry is not None and sorted(entry.files) == ["app.py", "old.txt", "src/deep/new.py"]

    result = cp.undo()
    assert result is not None and result.applied and not result.conflicts
    assert (ws / "app.py").read_text() == "v1\n" and (ws / "old.txt").read_text() == "keep me\n"
    assert not (ws / "src").exists()  # folders that held only created files are gone
    assert sorted(result.restored) == ["app.py", "old.txt"] and result.deleted == ["src/deep/new.py"]
    assert cp.entries() == ([], [entry])

    redone = cp.redo()
    assert redone is not None and redone.applied
    assert (ws / "app.py").read_text() == "v2\n" and not (ws / "old.txt").exists()
    assert (ws / "src" / "deep" / "new.py").read_text() == "new\n"
    assert cp.undo() is not None and cp.undo() is None  # one entry only


def test_no_change_no_entry_and_new_change_clears_redo(ws: Path) -> None:
    _write(ws / "a.txt", "1")
    cp = _store(ws)
    assert _turn(cp, lambda: None) is None
    _turn(cp, lambda: _write(ws / "a.txt", "2"))
    assert cp.undo() is not None
    _turn(cp, lambda: _write(ws / "b.txt", "x"))
    undo, redo = cp.entries()
    assert len(undo) == 1 and redo == []


def test_later_edits_block_undo_until_forced_and_unrelated_edits_survive(ws: Path) -> None:
    _write(ws / "a.txt", "agent-before")
    _write(ws / "notes.txt", "mine")
    cp = _store(ws)
    _turn(cp, lambda: _write(ws / "a.txt", "agent-after"))
    _write(ws / "notes.txt", "mine, edited later")  # unrelated: must survive the undo
    _write(ws / "a.txt", "hand-edited after the turn")

    blocked = cp.undo()
    assert blocked is not None and not blocked.applied and blocked.conflicts == ["a.txt"]
    assert (ws / "a.txt").read_text() == "hand-edited after the turn"  # nothing touched

    forced = cp.undo(force=True)
    assert forced is not None and forced.applied
    assert (ws / "a.txt").read_text() == "agent-before"
    assert (ws / "notes.txt").read_text() == "mine, edited later"
    refs = cp._git("for-each-ref", "--format=%(refname)").stdout.decode()
    assert f"refs/kartrix/{forced.entry.id}/overwritten" in refs  # the overridden state is kept


def test_users_own_git_repo_is_untouched(ws: Path) -> None:
    _write(ws / "a.txt", "committed\n")
    _user_git(ws, "init", "-q")
    _user_git(ws, "add", "a.txt")
    _user_git(ws, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    _write(ws / "a.txt", "work in progress\n")  # an unstaged change of the user
    head = _user_git(ws, "rev-parse", "HEAD")
    index = (ws / ".git" / "index").read_bytes()
    status = _user_git(ws, "status", "--porcelain")

    cp = _store(ws)
    _turn(cp, lambda: _write(ws / "b.txt", "agent"))
    cp.undo()

    assert _user_git(ws, "rev-parse", "HEAD") == head
    assert (ws / ".git" / "index").read_bytes() == index
    assert _user_git(ws, "status", "--porcelain") == status
    assert _user_git(ws, "stash", "list") == "" and _user_git(ws, "branch", "--list").strip() in ("* main", "* master")
    tree = cp._git("ls-tree", "-r", "--name-only", cp.snapshot()).stdout.decode()
    assert ".git" not in tree


def test_ignored_and_secret_files_are_never_snapshotted_or_touched(ws: Path) -> None:
    _write(ws / ".gitignore", "build/\n")
    _write(ws / ".env", "API_KEY=secret\n")
    _write(ws / ".env.example", "API_KEY=\n")
    cp = _store(ws)

    def change() -> None:
        _write(ws / "build" / "out.js", "generated")
        _write(ws / ".env", "API_KEY=changed\n")
        _write(ws / "node_modules" / "x" / "index.js", "dep")
        _write(ws / "app.py", "code")

    entry = _turn(cp, change)
    assert entry is not None and entry.files == ["app.py"]
    tree = cp._git("ls-tree", "-r", "--name-only", entry.after).stdout.decode().split()
    assert ".env" not in tree and ".env.example" in tree and "node_modules/x/index.js" not in tree
    cp.undo()
    assert not (ws / "app.py").exists()
    assert (ws / "build" / "out.js").exists() and (ws / ".env").read_text() == "API_KEY=changed\n"


def test_bytes_are_kept_exactly_despite_gitattributes(ws: Path) -> None:
    _write(ws / ".gitattributes", "* text=auto eol=crlf\n")
    lf, crlf = b"line1\nline2\n", b"one\r\ntwo\r\n"
    _write(ws / "lf.txt", lf)
    _write(ws / "crlf.txt", crlf)
    cp = _store(ws)

    def change() -> None:
        _write(ws / "lf.txt", b"changed\n")
        _write(ws / "crlf.txt", b"changed\r\n")

    _turn(cp, change)
    cp.undo()
    assert (ws / "lf.txt").read_bytes() == lf and (ws / "crlf.txt").read_bytes() == crlf


def test_workspace_attributes_cannot_make_git_run_a_filter(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "filter-ran"
    command = f"{Path(sys.executable).as_posix()} -c \\\"open(r'{marker.as_posix()}', 'w')\\\""
    evil = tmp_path / "evil.gitconfig"
    evil.write_text(f'[filter "evil"]\n\tclean = "{command}"\n', encoding="utf-8")
    _write(ws / ".gitattributes", "*.txt filter=evil\n")
    _write(ws / "a.txt", "data")
    assert GIT
    # Control: plain git with that config does run the filter.
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, "GIT_CONFIG_GLOBAL": str(evil)}
    subprocess.run([GIT, "init", "-q"], cwd=ws, env=env, check=True)
    subprocess.run([GIT, "hash-object", "a.txt"], cwd=ws, env=env, capture_output=True, check=False)
    assert marker.exists()
    marker.unlink()

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(evil))  # e.g. the user's global config defines the filter
    cp = _store(ws)
    _turn(cp, lambda: _write(ws / "a.txt", "changed"))
    cp.undo()
    assert not marker.exists()
    assert (ws / "a.txt").read_text() == "data"


def test_only_the_newest_entries_are_kept(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.checkpoints, "keep", 2)
    cp = _store(ws)
    entries = [_turn(cp, lambda i=i: _write(ws / f"f{i}.txt", "x"), f"t{i}") for i in range(3)]
    undo, _ = cp.entries()
    assert [e.label for e in undo] == ["t1", "t2"]
    refs = cp._git("for-each-ref", "--format=%(refname)").stdout.decode()
    assert entries[0] is not None and entries[0].id not in refs


async def test_track_records_changes_even_when_the_run_fails(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cp = _store(ws)
    monkeypatch.setattr(checkpoints, "_store", cp)
    async with checkpoints.track("/ask nothing"):
        pass
    with pytest.raises(RuntimeError):
        async with checkpoints.track("/ask write", session_id="s1"):
            _write(ws / "half.py", "partial")
            raise RuntimeError("stopped mid-way")
    undo, _ = cp.entries()
    assert [(e.label, e.session_id, e.files) for e in undo] == [("/ask write", "s1", ["half.py"])]


async def test_track_without_a_store_just_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "_store", None)
    ran = []
    async with checkpoints.track("x"):
        ran.append(1)
    assert ran == [1]


def test_setup_reports_missing_git(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "find_executable", lambda *a: None)
    assert "git was not found" in (checkpoints.setup() or "")
    assert checkpoints.current() is None
    monkeypatch.undo()
    monkeypatch.setenv("KARTRIX_HOME", str(ws.parent / "home"))
    assert checkpoints.setup() is None and checkpoints.current() is not None
    monkeypatch.setattr(checkpoints, "_store", None)  # don't leak the store into other tests
