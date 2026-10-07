"""Eval results on disk, stamped with everything that decides them.

Each run writes ``results.json`` (and ``report.html``) to ``.kartrix/evals/results/<time>-<suite>/``.
The ``versions`` block says what was measured: Kartrix version and git commit (dirty or not), main /
judge / embedding models, a hash of the prompts, a hash of the effective configuration and a hash of
each dataset — two results are only comparable when the datasets match. ``--save-baseline`` copies a
result to ``evals/baselines/<suite>.json`` (committed), the reference ``--compare`` checks against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kartrix.config import PROJECT_ROOT, settings
from kartrix.evals.datasets import dataset_version, evals_dir
from kartrix.evals.fixtures import cache_dir

SUITES = ("rag", "agent", "calibrate")
_DATASETS = {"rag": ("repos.yaml", "rag/golden.yaml"), "agent": ("agent",), "calibrate": ("rag/calibration.yaml",)}


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False)  # noqa: S603, S607
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def prompt_version() -> str:
    """Hash of every prompt that shapes what is measured."""
    from kartrix.agent.factory import SYSTEM_PROMPT
    from kartrix.evals.rag import ANSWER_PROMPT
    from kartrix.tasks import executor, planner

    texts = [SYSTEM_PROMPT, planner._SYSTEM_PROMPT, executor._JUDGE_SYSTEM_PROMPT, ANSWER_PROMPT]
    return hashlib.sha256("\n\x00".join(texts).encode()).hexdigest()[:12]


def config_version() -> str:
    """Hash of the effective settings (they hold no secrets)."""
    data = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def versions(suites: list[str]) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    from kartrix.sandbox.manager import get_sandbox

    try:
        kartrix_version = version("kartrix")
    except PackageNotFoundError:
        kartrix_version = "unknown"
    try:
        deepeval_version = version("deepeval")
    except PackageNotFoundError:
        deepeval_version = None
    sandbox = get_sandbox()
    return {
        "kartrix": kartrix_version,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model": f"{settings.llm.provider}:{settings.llm.model}",
        "judge_model": f"{settings.llm.provider}:{settings.llm.effective_judge_model}",
        "embeddings": f"{settings.embeddings.provider}:{settings.embeddings.model}",
        "retrieval": settings.retrieval.model_dump(mode="json"),
        "prompts": prompt_version(),
        "config": config_version(),
        "datasets": {name: dataset_version(*_DATASETS[name]) for name in suites},
        "deepeval": deepeval_version,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "sandbox": sandbox.name if sandbox is not None else None,
    }


def results_dir() -> Path:
    path = cache_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_dir(suite: str, out: str | None = None) -> Path:
    path = Path(out) if out else results_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{suite}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def baseline_path(suite: str) -> Path:
    return evals_dir() / "baselines" / f"{suite}.json"


def load_baseline(suite: str) -> dict[str, Any] | None:
    path = baseline_path(suite)
    return load_json(path) if path.is_file() else None


def latest_results(suite: str) -> Path | None:
    runs = sorted(results_dir().glob(f"*-{suite}/results.json"))
    return runs[-1] if runs else None
