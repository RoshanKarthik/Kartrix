"""The eval datasets (``evals/`` in the repository), loaded strictly like the config.

Layout::

    evals/repos.yaml                 fixture repositories for the RAG questions (pinned commits)
    evals/rag/golden.yaml            golden questions, each labelled with the files/symbols that answer it
    evals/rag/calibration.yaml       hand-labelled (question, context, answer) items to calibrate the judge
    evals/agent/apps/<app>/          small vendored apps the agent tasks run on
    evals/agent/tasks/<id>/task.yaml one coding task with its executable check
    evals/agent/tasks/<id>/setup/    files added to the app for this task only (e.g. a planted injection)
    evals/agent/tasks/<id>/hidden/   files copied into the workspace *after* the run, before the check
                                     (tests the agent never saw, so it can't fit the code to them)
    evals/agent/tasks/<id>/solution/ a reference solution: the dataset tests prove the check fails on the
                                     untouched app and passes with it
    evals/agent/shared/              check helpers copied to ``.eval/`` in every workspace before the checks
                                     (e.g. ``mutation_check.py``)
    evals/baselines/<suite>.json     the recorded baseline results (``--save-baseline``)
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kartrix.config import PROJECT_ROOT, BudgetLimits, ConfigError, load_yaml_strict
from kartrix.headless.spec import ApprovalPolicy


def evals_dir() -> Path:
    """``evals/`` of this checkout (``KARTRIX_EVALS_DIR`` points elsewhere, e.g. in tests)."""
    return Path(os.environ.get("KARTRIX_EVALS_DIR") or PROJECT_ROOT / "evals")


class DatasetError(Exception):
    """A dataset file is missing or invalid. Safe to show."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")


def _check_id(value: str) -> str:
    if not _ID.match(value):
        raise ValueError(f"{value!r} is not a valid id (lowercase letters, digits, - _ .)")
    return value


# ── fixture repositories ──────────────────────────────────────────────


class FixtureRepo(_Model):
    name: str
    # self: this repository at `commit` (git archive) · git: cloned from `url` at `commit`
    source: Literal["self", "git"]
    url: str | None = None
    commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    stacks: list[Literal["python", "typescript"]]
    description: str = ""

    @model_validator(mode="after")
    def _url_for_git(self) -> FixtureRepo:
        if self.source == "git" and not (self.url or "").startswith("https://"):
            raise ValueError(f"repo {self.name}: a git source needs an https:// url")
        return self

    @property
    def key(self) -> str:
        """Folder name in the cache: the commit is part of it, so a new pin never reuses an old checkout."""
        return f"{self.name}@{self.commit[:12]}"


class ReposFile(_Model):
    repos: list[FixtureRepo]


# ── RAG ───────────────────────────────────────────────────────────────

QuestionKind = Literal["locate", "explain", "flow", "config", "security", "test"]


class GoldenQuestion(_Model):
    id: str
    repo: str
    question: str
    kind: QuestionKind
    files: list[str] = Field(min_length=1)  # repo-relative paths (posix) that answer the question
    symbols: list[str] = Field(default_factory=list)  # functions/classes/methods that answer it
    answer: str  # short reference answer (expected output for the context recall judge)
    quick: bool = False  # part of the fast everyday subset

    _id = field_validator("id")(_check_id)


class GoldenSet(_Model):
    questions: list[GoldenQuestion]


class CalibrationItem(_Model):
    """One hand-labelled judgement: does ``answer`` follow from ``context`` / address ``question``?"""

    id: str
    question: str
    context: list[str] = Field(min_length=1)
    answer: str
    faithful: bool  # every claim in the answer is supported by the context
    relevant: bool  # the answer addresses the question
    note: str = ""

    _id = field_validator("id")(_check_id)


class CalibrationSet(_Model):
    items: list[CalibrationItem]


# ── agent tasks ───────────────────────────────────────────────────────

TaskCategory = Literal["bugfix", "feature", "test", "refactor", "explain", "locate", "safety", "budget"]


class CheckStep(_Model):
    """A command run in the workspace after the agent finished. ``{python}`` and ``{node}`` in the
    arguments become the interpreter running the evals and the ``node`` on PATH."""

    run: list[str] = Field(min_length=1)
    expect: Literal["pass", "fail"] = "pass"  # fail: e.g. the agent's new tests must catch a planted bug
    timeout: float = Field(120, gt=0)
    name: str = ""


class AnswerCheck(_Model):
    """For explain/locate tasks: the final answer must mention these (case-insensitive)."""

    contains_all: list[str] = Field(default_factory=list)
    contains_any: list[str] = Field(default_factory=list)
    reference: str = ""  # a correct answer, for the judge's answer-correctness score


class Expectations(_Model):
    """What a good run looks like besides the checks. Hard expectations decide success; ``tools``
    and ``approval_for`` are scored as metrics only (there is more than one good way)."""

    status: Literal["completed", "stopped"] = "completed"
    tools: list[str] = Field(default_factory=list)  # tools a good trajectory uses (tool correctness)
    approval_for: list[str] = Field(default_factory=list)  # command patterns that should be asked about
    blocked: list[str] = Field(default_factory=list)  # command patterns that must never run
    forbidden_paths: list[str] = Field(default_factory=list)  # must not exist afterwards (injection canaries)
    unchanged: list[str] = Field(default_factory=list)  # glob patterns of files that must not change
    within_budget: bool = True  # usage must stay within the run's limits


class AgentTask(_Model):
    id: str
    app: str
    stack: Literal["python", "typescript"]
    category: TaskCategory
    task: str
    mode: Literal["ask", "plan"] = "ask"
    permissions: Literal["read_only", "default", "auto"] = "default"
    budget: BudgetLimits | None = None  # fields given replace budgets.turn / budgets.plan for this run
    approvals: ApprovalPolicy = ApprovalPolicy()
    check: list[CheckStep] = Field(default_factory=list)
    answer: AnswerCheck | None = None
    expect: Expectations = Expectations()
    quick: bool = False
    timeout: float = Field(1200, gt=0)  # the whole `kartrix run`, seconds
    notes: str = ""  # why the task exists / what it probes

    _id = field_validator("id")(_check_id)

    @model_validator(mode="after")
    def _has_a_check(self) -> AgentTask:
        if not self.check and self.answer is None and self.category not in ("safety", "budget"):
            raise ValueError(f"task {self.id}: needs `check` steps or an `answer` check")
        return self

    @property
    def folder(self) -> Path:
        return evals_dir() / "agent" / "tasks" / self.id

    @property
    def setup_dir(self) -> Path:
        return self.folder / "setup"

    @property
    def app_dir(self) -> Path:
        return evals_dir() / "agent" / "apps" / self.app


# ── loading ───────────────────────────────────────────────────────────


def _load[M: BaseModel](path: Path, model: type[M]) -> M:
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    try:
        return model.model_validate(load_yaml_strict(path))
    except (ConfigError, ValidationError) as e:
        raise DatasetError(f"invalid dataset {path}:\n{e}") from None


def _unique(ids: list[str], what: str) -> None:
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DatasetError(f"duplicate {what} ids: {', '.join(dupes)}")


def load_repos() -> dict[str, FixtureRepo]:
    repos = _load(evals_dir() / "repos.yaml", ReposFile).repos
    _unique([r.name for r in repos], "repo")
    return {r.name: r for r in repos}


def load_golden(quick: bool = False, ids: list[str] | None = None) -> list[GoldenQuestion]:
    questions = _load(evals_dir() / "rag" / "golden.yaml", GoldenSet).questions
    _unique([q.id for q in questions], "question")
    repos = load_repos()
    unknown = sorted({q.repo for q in questions} - set(repos))
    if unknown:
        raise DatasetError(f"golden questions refer to unknown repos: {', '.join(unknown)}")
    if ids:
        missing = sorted(set(ids) - {q.id for q in questions})
        if missing:
            raise DatasetError(f"no such question: {', '.join(missing)}")
        return [q for q in questions if q.id in ids]
    return [q for q in questions if q.quick] if quick else questions


def load_calibration() -> list[CalibrationItem]:
    items = _load(evals_dir() / "rag" / "calibration.yaml", CalibrationSet).items
    _unique([i.id for i in items], "calibration item")
    return items


def load_tasks(quick: bool = False, ids: list[str] | None = None) -> list[AgentTask]:
    root = evals_dir() / "agent" / "tasks"
    tasks = [_load(p, AgentTask) for p in sorted(root.glob("*/task.yaml"))]
    _unique([t.id for t in tasks], "task")
    for t in tasks:
        if t.folder.name != t.id:
            raise DatasetError(f"task {t.id} lives in folder {t.folder.name}; the folder must be named after the id")
        if not t.app_dir.is_dir():
            raise DatasetError(f"task {t.id}: no app {t.app} in {t.app_dir.parent}")
    if ids:
        missing = sorted(set(ids) - {t.id for t in tasks})
        if missing:
            raise DatasetError(f"no such task: {', '.join(missing)}")
        return [t for t in tasks if t.id in ids]
    return [t for t in tasks if t.quick] if quick else tasks


def dataset_version(*parts: str) -> str:
    """Hash of the dataset files under ``evals/<part>`` — results from different versions aren't comparable."""
    digest = hashlib.sha256()
    for part in parts:
        base = evals_dir() / part
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for p in files:
            if "__pycache__" in p.parts or "node_modules" in p.parts:
                continue
            digest.update(p.relative_to(evals_dir()).as_posix().encode())
            digest.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]
