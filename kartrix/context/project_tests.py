"""Does the project have tests? A file-name and config check, for the reviewer.

The reviewer must run the project's tests in every review. In a project with none it used to keep globbing for
test files instead (live, 2026-10-20: a workspace holding only ``calc.py``, ``**/test*.py`` several times across
rounds) and never gave a verdict. The graph now checks once, before the review, and tells the reviewer what it
found: the test files and test commands, or that there are none, so it checks the change another way.

The check errs towards "has tests": any test-looking file or directory, or a config that defines a test command,
counts, and a walk cut short by the file limit is "unknown", never "none".
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from kartrix.context.discovery import RepoFilter, iter_files
from kartrix.observability.logger import get_logger
from kartrix.security.workspace import CASE_INSENSITIVE, get_workspace

logger = get_logger(__name__)

_MAX_FILES = 20_000  # a bigger tree without a hit is "unknown": the reviewer then works as before
_MAX_EVIDENCE = 5
_MAX_CONFIG_BYTES = 256 * 1024

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "testing"})
_TEST_FILE = re.compile(
    r"^(?:test_.*\.py|.*_test\.py|tests?\.py|conftest\.py"  # Python
    r"|.*\.(?:test|spec)\.[cm]?[jt]sx?"  # JavaScript / TypeScript
    r"|.*_test\.go|.*_(?:spec|test)\.rb|.*Tests?\.(?:java|kt|cs|swift)|.*_test\.exs?|test_.*\.sh)$",
    re.IGNORECASE if CASE_INSENSITIVE else 0,
)
# Config files that define a test command. Cargo always has one ("cargo test" runs the #[cfg(test)] modules).
_TEST_CONFIGS = frozenset({"pytest.ini", "tox.ini", "noxfile.py", "Cargo.toml"})
_MAKE_TARGET = re.compile(r"^(?:test|tests|check)\s*:", re.MULTILINE)
_NPM_DEFAULT_TEST = "no test specified"  # "npm init" writes a test script that only fails


@dataclass(frozen=True)
class TestSurvey:
    """What the check found. ``evidence``: test files and test commands (the first few); ``complete``: the whole
    workspace was checked. ``none`` only when both say so."""

    __test__ = False  # not a pytest test class, despite the name

    evidence: tuple[str, ...] = ()
    complete: bool = False

    @property
    def none(self) -> bool:
        return self.complete and not self.evidence


def _read(path: Path) -> str:
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _test_command(path: Path, rel: str) -> str | None:
    """Where ``path`` defines a test command, e.g. ``package.json: scripts.test``; None if it doesn't."""
    name = path.name
    if name in _TEST_CONFIGS:
        return rel
    try:
        if name == "package.json":
            script = (json.loads(_read(path) or "{}").get("scripts") or {}).get("test")
            return f"{rel}: scripts.test" if isinstance(script, str) and _NPM_DEFAULT_TEST not in script else None
        if name == "pyproject.toml":
            tool = tomllib.loads(_read(path)).get("tool") or {}
            found = [k for k in ("pytest", "tox", "nox") if k in tool]
            return f"{rel}: [tool.{found[0]}]" if found else None
        if name == "setup.cfg":
            return f"{rel}: [tool:pytest]" if "[tool:pytest]" in _read(path) else None
        if name in ("Makefile", "makefile", "GNUmakefile"):
            m = _MAKE_TARGET.search(_read(path))
            return f"{rel}: make {m.group(0).rstrip(': ')}" if m else None
    except (OSError, ValueError, AttributeError):  # unreadable or malformed: it defines nothing we can run
        return None
    return None


def survey_tests(max_files: int = _MAX_FILES) -> TestSurvey:
    """Look through the workspace (``.gitignore`` and the read deny-list apply, as for ``glob``) for test files
    and test commands; stops after the first few."""
    ws = get_workspace()
    evidence: list[str] = []
    seen = 0
    for path in iter_files(RepoFilter.load(ws.root)):
        if not ws.is_readable(path):
            continue
        seen += 1
        if seen > max_files:
            return TestSurvey(tuple(evidence), complete=False)
        rel = ws.relative(path)
        dirs = {d.lower() for d in rel.split("/")[:-1]}
        hit = rel if dirs & _TEST_DIRS or _TEST_FILE.match(path.name) else _test_command(path, rel)
        if hit:
            evidence.append(hit)
            if len(evidence) >= _MAX_EVIDENCE:
                break
    return TestSurvey(tuple(evidence), complete=True)


def review_note(survey: TestSurvey) -> str:
    """The section of the reviewer's brief that says what tests there are."""
    if survey.evidence:
        more = " (and maybe more)" if len(survey.evidence) >= _MAX_EVIDENCE else ""
        found = "\n".join(f"- {e}" for e in survey.evidence)
        return f"## Tests in this project\nTest files and test commands found{more}:\n{found}\nRun the tests."
    if survey.none:
        return (
            "## Tests in this project\nNone: Kartrix checked the whole workspace and found no test files and no test "
            "command (no test_*.py, *.test.ts, *_test.go or tests/ directory; no pytest/tox config, package.json "
            "test script or Makefile test target). Don't search for tests or run a test runner. Check the change "
            "another way — read the changed files and, where it helps, run the changed program or module with "
            'run_command (its file, e.g. "python calc.py", not "python -c") — then give your verdict.'
        )
    return ""
