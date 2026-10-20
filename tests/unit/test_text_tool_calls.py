"""The reviewer's verdict when the model ends its turn in text (kartrix.llm.nim, kartrix.agent.reliability).

Seen live (2026-10-20, Nemotron 3 Ultra on NIM): asked for a forced tool call, the model wrote the call into the
message text and stopped — ``finish_reason: stop``, no tool calls — so the reviewer finished "without a verdict"
and a correct change went back to the coder. The fake models below reply exactly like that.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from kartrix.agent import graph as graph_mod
from kartrix.agent.factory import agent_middleware
from kartrix.agent.graph import REVIEWER_PROMPT, ReviewVerdict
from kartrix.agent.reliability import CompletionGuardMiddleware, answer_tool_nudge, is_nudge
from kartrix.config import settings
from kartrix.llm.factory import response_format_for
from kartrix.llm.nim import RecoverTextToolCalls, parse_text_tool_calls, recover_tool_calls
from kartrix.security.approvals import ApprovalMiddleware

# Two of the replies NIM returned under tool_choice="required" (brackets as sent).
LIVE_GLOBS = (
    '[[{ "name": "glob", "parameters": { "pattern": "*test*" } }, '
    '{ "name": "glob", "parameters": { "pattern": "pytest.ini" } }]]'
)
LIVE_GLOB_UNBALANCED = '[[{\n  "name": "glob",\n  "parameters": {"pattern": "**/*test*.py"}\n} \n]'
VERDICT_AS_TEXT = '[[{"name": "ReviewVerdict", "parameters": {"approved": true, "issues": []}}]]'
TOOLS = {"glob", "read_file", "ReviewVerdict"}


def _stop(content: str) -> AIMessage:
    return AIMessage(content=content, response_metadata={"finish_reason": "stop"})


# ── parsing ─────────────────────────────────────────────────────────


def test_live_replies_parse_into_tool_calls() -> None:
    calls = parse_text_tool_calls(LIVE_GLOBS, TOOLS)
    assert calls is not None
    assert [(c["name"], c["args"]) for c in calls] == [
        ("glob", {"pattern": "*test*"}),
        ("glob", {"pattern": "pytest.ini"}),
    ]
    assert len({c["id"] for c in calls}) == 2
    calls = parse_text_tool_calls(LIVE_GLOB_UNBALANCED, TOOLS)
    assert calls is not None and calls[0]["args"] == {"pattern": "**/*test*.py"}


def test_other_spellings_of_a_call() -> None:
    tagged = '<TOOLCALL>[{"name": "read_file", "arguments": "{\\"file_path\\": \\"calc.py\\"}"}]</TOOLCALL>'
    calls = parse_text_tool_calls(tagged, TOOLS)
    assert calls is not None and calls[0]["args"] == {"file_path": "calc.py"}
    calls = parse_text_tool_calls('{"name": "ReviewVerdict", "arguments": {"approved": false, "issues": ["x"]}}', TOOLS)
    assert calls is not None and calls[0]["args"] == {"approved": False, "issues": ["x"]}


def test_text_that_is_not_only_calls_to_bound_tools_stays_text() -> None:
    for text in (
        "The change looks correct.",
        'I will now call {"name": "glob", "parameters": {"pattern": "*"}}',  # prose around it
        '[{"name": "delete_everything", "parameters": {}}]',  # not a bound tool
        '[{"name": "glob", "parameters": ["*"]}]',  # arguments not an object
        '[{"name": "glob", "parameters": {"pattern": "*"}',  # cut off
        '[{"route": "change"}]',  # JSON, but no call
        "[]",
        "",
    ):
        assert parse_text_tool_calls(text, TOOLS) is None, text


def test_recovery_leaves_real_tool_calls_and_unbound_models_alone() -> None:
    real = AIMessage(content=LIVE_GLOBS, tool_calls=[{"name": "glob", "args": {"pattern": "a"}, "id": "c1"}])
    assert recover_tool_calls(real, TOOLS) is real
    text = _stop(LIVE_GLOBS)
    assert recover_tool_calls(text, set()) is text  # no tools bound: a JSON answer is just an answer
    fixed = recover_tool_calls(text, TOOLS)
    assert fixed.content == "" and [c["name"] for c in fixed.tool_calls] == ["glob", "glob"]
    assert fixed.response_metadata == {"finish_reason": "stop"}


# ── the reviewer loop ───────────────────────────────────────────────


class _Scripted(GenericFakeChatModel):
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class _NimLike(RecoverTextToolCalls, _Scripted):
    """Scripted like ``_Scripted``, with the recovery ``kartrix.llm.nim.ChatNVIDIA`` has."""


@tool
def glob(pattern: str) -> str:
    """Find files by glob pattern."""
    return "(no matches)"


@tool
def read_file(file_path: str) -> str:
    """Read a file."""
    return "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"


def _reviewer(model: GenericFakeChatModel, answer_tool: str | None = ReviewVerdict.__name__) -> Any:
    return create_agent(
        model, tools=[read_file, glob], system_prompt=REVIEWER_PROMPT,
        middleware=[CompletionGuardMiddleware(max_nudges=2, answer_tool=answer_tool)],
        response_format=response_format_for(ReviewVerdict),
    )  # fmt: skip


def _read_call() -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "calc.py"}, "id": "r1"}])


async def _review(agent: Any) -> dict[str, Any]:
    prompt = "Request: Add a function subtract(a, b) to calc.py that returns a - b.\n\nCheck the change and give your verdict."
    out: dict[str, Any] = await agent.ainvoke({"messages": [HumanMessage(prompt)]})
    return out


async def test_bare_stop_without_the_fixes_leaves_no_verdict() -> None:
    """The failure as it was: the loop ends on the text reply and the reviewer has no verdict."""
    model = _Scripted(messages=iter([_read_call(), _stop(LIVE_GLOB_UNBALANCED)]))
    out = await _review(_reviewer(model, answer_tool=None))
    assert out.get("structured_response") is None


async def test_calls_written_as_text_are_run_and_the_verdict_is_approving() -> None:
    model = _NimLike(messages=iter([_read_call(), _stop(LIVE_GLOB_UNBALANCED), _stop(VERDICT_AS_TEXT)]))
    out = await _review(_reviewer(model))
    assert out["structured_response"] == ReviewVerdict(approved=True, issues=[])
    assert any(isinstance(m, ToolMessage) and m.name == "glob" and m.content == "(no matches)" for m in out["messages"])
    assert not any(is_nudge(m) for m in out["messages"])  # recovered, nothing to nudge


async def test_prose_answer_is_nudged_to_the_verdict_tool() -> None:
    verdict_call = AIMessage(
        content="", tool_calls=[{"name": "ReviewVerdict", "args": {"approved": True, "issues": []}, "id": "v1"}]
    )
    model = _Scripted(messages=iter([_read_call(), _stop("subtract is correct and there are no tests."), verdict_call]))
    out = await _review(_reviewer(model))
    assert out["structured_response"] == ReviewVerdict(approved=True, issues=[])
    nudge = model.seen[2][-1]
    assert is_nudge(nudge) and nudge.content == answer_tool_nudge("ReviewVerdict")
    assert model.seen[2][-2].content == "subtract is correct and there are no tests."  # the reasoning is kept


async def test_nudges_run_out_without_a_verdict() -> None:
    model = _Scripted(messages=iter([_stop("Looks fine."), _stop("Still fine."), _stop("Fine.")]))
    out = await _review(_reviewer(model))
    assert out.get("structured_response") is None
    assert sum(is_nudge(m) for m in out["messages"]) == 2


# ── the graph's review step ─────────────────────────────────────────


def _graph(monkeypatch: pytest.MonkeyPatch, main: GenericFakeChatModel) -> graph_mod.AgentGraph:
    monkeypatch.setattr(settings.llm, "fallbacks", [])
    monkeypatch.setattr(graph_mod, "get_chat_model", lambda role="main", **kw: main)
    return graph_mod.AgentGraph(middleware=agent_middleware, approval=ApprovalMiddleware)


def _state() -> graph_mod.AgentState:
    return {
        "messages": [HumanMessage("Add a function subtract(a, b) to calc.py that returns a - b.")],
        "change_report": "Added subtract to calc.py.",
        "changed_files": ["calc.py"],
        "rounds": 1,
    }


async def test_review_step_approves_a_correct_change_on_the_first_review(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _NimLike(messages=iter([_read_call(), _stop(LIVE_GLOB_UNBALANCED), _stop(VERDICT_AS_TEXT)]))
    update = await _graph(monkeypatch, model).review(_state())
    assert update["approved"] is True and update["review_notes"] == ""
    assert graph_mod.AgentGraph.after_review({**_state(), **update}) == "respond"


async def test_review_step_without_a_verdict_passes_on_what_the_reviewer_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Scripted(messages=iter([_stop("calc.py has no tests."), _stop("No tests."), _stop("No tests at all.")]))
    update = await _graph(monkeypatch, model).review(_state())
    assert update["approved"] is False
    assert update["review_notes"] == "No tests at all."  # not "no verdict (stopped)": the run wasn't stopped
