"""The eval datasets are valid, and every agent task is meaningful: its checks fail on the app as the
agent finds it (setup/ applied) and pass with the reference solution."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kartrix.evals.agent import run_check
from kartrix.evals.datasets import evals_dir, load_calibration, load_golden, load_repos, load_tasks
from kartrix.evals.fixtures import copy_into, prepare_workspace

TASKS = load_tasks()


def test_golden_set_is_valid():
    questions = load_golden()
    repos = load_repos()
    assert len(questions) >= 100
    assert {q.repo for q in questions} == set(repos)
    assert len(load_golden(quick=True)) >= 10
    for q in questions:
        assert all("\\" not in f and not f.startswith("/") for f in q.files), q.id


def test_kartrix_labels_exist_at_the_pinned_commit():
    """The self fixture is this repository at a commit: its labelled files must exist there."""
    import subprocess

    repo = load_repos()["kartrix"]
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", repo.commit], capture_output=True, text=True, check=False
    )
    if listing.returncode != 0:
        pytest.skip("the pinned commit is not in this clone")
    files = set(listing.stdout.split())
    missing = [(q.id, f) for q in load_golden() if q.repo == "kartrix" for f in q.files if f not in files]
    assert not missing


def test_calibration_set_is_balanced():
    items = load_calibration()
    assert len(items) >= 20
    assert any(i.faithful for i in items) and any(not i.faithful for i in items)
    assert any(i.relevant for i in items) and any(not i.relevant for i in items)


def test_tasks_cover_every_category_and_stack():
    assert len(TASKS) >= 25
    assert {t.category for t in TASKS} >= {"bugfix", "feature", "test", "refactor", "explain", "locate", "safety"}
    assert {t.stack for t in TASKS} == {"python", "typescript"}
    assert any(t.mode == "plan" for t in TASKS)
    assert 4 <= sum(t.quick for t in TASKS) <= 10


def _workspace(task, tmp_path: Path, with_solution: bool) -> Path:
    ws = prepare_workspace(task.app_dir, tmp_path / "ws", task.setup_dir)
    if with_solution:
        copy_into(task.folder / "solution", ws)
    copy_into(evals_dir() / "agent" / "shared", ws / ".eval")
    if (task.folder / "hidden").is_dir():
        copy_into(task.folder / "hidden", ws)
    return ws


@pytest.mark.parametrize("task", [t for t in TASKS if t.check], ids=lambda t: t.id)
def test_task_check_fails_before_and_passes_with_solution(task, tmp_path):
    if task.stack == "typescript" and shutil.which("node") is None:
        pytest.skip("node is not installed")
    assert (task.folder / "solution").is_dir(), "a task with checks needs a reference solution"
    before = [run_check(step, _workspace(task, tmp_path / "before", False)) for step in task.check]
    assert not all(c["ok"] for c in before), "the checks already pass on the untouched app: the task is trivial"
    after = [run_check(step, _workspace(task, tmp_path / "after", True)) for step in task.check]
    failed = [c for c in after if not c["ok"]]
    assert not failed, failed[0]["output_tail"] if failed else ""
