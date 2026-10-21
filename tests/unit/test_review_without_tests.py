"""The reviewer in a project without tests (kartrix.context.project_tests, AgentGraph.review).

Seen live (2026-10-20, /ask "Add a function subtract(a, b) to calc.py" in a workspace holding only calc.py): told
to run the tests in every review, the reviewer kept globbing for test files, round after round, instead of giving a
verdict. The review step now checks the workspace first and puts what it found into the reviewer's brief.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from kartrix.agent import graph as graph_mod
from kartrix.agent.factory import agent_middleware
from kartrix.agent.graph import CODER_PROMPT, COMMAND_RULE, REVIEWER_PROMPT
from kartrix.config import settings
from kartrix.context.project_tests import TestSurvey, review_note, survey_tests
from kartrix.security import workspace as ws_mod
from kartrix.security.approvals import ApprovalMiddleware
from kartrix.security.command_policy import evaluate
from kartrix.security.workspace import Workspace

CALC = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace holding only calc.py, as in the live run."""
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    monkeypatch.setattr(ws_mod, "_current", Workspace.create(tmp_path))
    return tmp_path


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── the check ───────────────────────────────────────────────────────


def test_a_project_without_tests_is_none(root: Path) -> None:
    _write(root, "README.md", "# calc\n")
    _write(root, "pyproject.toml", '[project]\nname = "calc"\ndependencies = []\n')
    _write(root, "package.json", '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1"}}')
    _write(root, "Makefile", "build:\n\tpython -m build\n")
    survey = survey_tests()
    assert survey == TestSurvey((), complete=True) and survey.none


@pytest.mark.parametrize(
    ("rel", "text", "evidence"),
    [
        ("tests/test_calc.py", "", "tests/test_calc.py"),
        ("test_calc.py", "", "test_calc.py"),
        ("calc_test.py", "", "calc_test.py"),
        ("src/calc.test.ts", "", "src/calc.test.ts"),
        ("web/__tests__/calc.js", "", "web/__tests__/calc.js"),
        ("calc_test.go", "", "calc_test.go"),
        ("src/test/java/CalcTest.java", "", "src/test/java/CalcTest.java"),
        ("pytest.ini", "[pytest]\n", "pytest.ini"),
        ("pyproject.toml", "[tool.pytest.ini_options]\naddopts = '-q'\n", "pyproject.toml: [tool.pytest]"),
        ("setup.cfg", "[tool:pytest]\n", "setup.cfg: [tool:pytest]"),
        ("package.json", '{"scripts": {"test": "node --test"}}', "package.json: scripts.test"),
        ("Makefile", "test:\n\tpython -m unittest\n", "Makefile: make test"),
        ("Cargo.toml", '[package]\nname = "calc"\n', "Cargo.toml"),
    ],
)
def test_test_files_and_test_commands_count(root: Path, rel: str, text: str, evidence: str) -> None:
    _write(root, rel, text)
    survey = survey_tests()
    assert survey.evidence == (evidence,) and survey.complete and not survey.none


def test_ignored_and_malformed_files_say_nothing(root: Path) -> None:
    _write(root, ".gitignore", "vendor/\n")
    _write(root, "vendor/lib/test_lib.py")  # someone else's tests, ignored
    _write(root, "node_modules/left-pad/test.js")  # excluded by config
    _write(root, "pyproject.toml", "[tool.pytest\n")  # not TOML
    _write(root, "package.json", "{")
    assert survey_tests().none


def test_the_check_stops_after_a_few_hits(root: Path) -> None:
    for i in range(10):
        _write(root, f"tests/test_{i}.py")
    assert len(survey_tests().evidence) == 5


def test_a_tree_too_big_to_check_is_unknown_not_none(root: Path) -> None:
    for i in range(5):
        _write(root, f"src/mod_{i}.py")
    survey = survey_tests(max_files=3)
    assert not survey.complete and not survey.evidence and not survey.none
    assert review_note(survey) == ""  # nothing to say: the reviewer works as before


def test_the_notes() -> None:
    none = review_note(TestSurvey((), complete=True))
    assert "found no test files and no test command" in none and "Don't search for tests" in none
    found = review_note(TestSurvey(("tests/test_calc.py",), complete=True))
    assert "- tests/test_calc.py" in found and "Run the tests." in found and "no test" not in found


def test_the_reviewer_runs_files_not_inline_code(root: Path) -> None:
    """Live (2026-10-21) the reviewer checked calc.py with `python -c "import calc; …"`: the policy can't tell what
    inline code does, so even in auto mode the user is asked. Running the file is a plain run."""
    assert COMMAND_RULE in REVIEWER_PROMPT and COMMAND_RULE in CODER_PROMPT
    assert '"python calc.py"' in REVIEWER_PROMPT and 'not "python -c"' in review_note(TestSurvey((), complete=True))
    assert evaluate('python -c "import calc"', mode="auto", log=False).action == "ask"
    assert evaluate('node -e "1"', mode="auto", log=False).action == "ask"
    assert evaluate("python calc.py", mode="auto", log=False).action == "allow"


def test_the_reviewer_keeps_the_rule_for_projects_with_tests() -> None:
    assert "never approve, or say the tests can't run, without having run them" in REVIEWER_PROMPT
    assert "Only when the\nreview request says the project has no tests and no test command" in REVIEWER_PROMPT


# ── the review step ─────────────────────────────────────────────────


class _Scripted(GenericFakeChatModel):
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _graph(monkeypatch: pytest.MonkeyPatch, main: GenericFakeChatModel) -> graph_mod.AgentGraph:
    monkeypatch.setattr(settings.llm, "fallbacks", [])
    monkeypatch.setattr(graph_mod, "get_chat_model", lambda role="main", **kw: main)
    return graph_mod.AgentGraph(middleware=agent_middleware, approval=ApprovalMiddleware)


def _state() -> graph_mod.AgentState:
    return {
        "messages": [HumanMessage("Add a function subtract(a, b) to calc.py that returns a - b.")],
        "change_report": "Added subtract(a, b) to calc.py. The project has no tests.",
        "changed_files": ["calc.py"],
        "rounds": 1,
    }


def _brief(model: _Scripted) -> str:
    first = model.seen[0][-1]
    assert isinstance(first, HumanMessage)
    return str(first.content)


async def test_without_tests_the_reviewer_reads_the_change_and_gives_a_verdict(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Scripted(
        messages=iter([
            _call("read_file", {"file_path": "calc.py"}, "r1"),
            _call("ReviewVerdict", {"approved": True, "issues": []}, "v1"),
        ])
    )  # fmt: skip
    update = await _graph(monkeypatch, model).review(_state())
    assert update["approved"] is True and update["review_notes"] == ""
    brief = _brief(model)
    assert "## Tests in this project\nNone:" in brief and "Don't search for tests" in brief
    assert brief.index("## Tests in this project") < brief.index("Check the change and give your verdict.")
    read = model.seen[1][-1]  # the real read_file ran in the workspace
    assert isinstance(read, ToolMessage) and "return a - b" in str(read.content)


async def test_with_tests_the_brief_lists_them_and_says_to_run_them(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        root,
        "tests/test_calc.py",
        "from calc import subtract\n\n\ndef test_subtract():\n    assert subtract(3, 1) == 2\n",
    )
    model = _Scripted(messages=iter([_call("ReviewVerdict", {"approved": False, "issues": ["tests not run"]}, "v1")]))
    update = await _graph(monkeypatch, model).review(_state())
    assert update["approved"] is False
    brief = _brief(model)
    assert "- tests/test_calc.py" in brief and "Run the tests." in brief and "None:" not in brief


async def test_a_failed_check_leaves_the_brief_as_it_was(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(max_files: int = 0) -> TestSurvey:
        raise OSError("disk gone")

    monkeypatch.setattr(graph_mod, "survey_tests", broken)
    model = _Scripted(messages=iter([_call("ReviewVerdict", {"approved": True, "issues": []}, "v1")]))
    await _graph(monkeypatch, model).review(_state())
    assert "## Tests in this project" not in _brief(model)
