"""Context assembly (kartrix.agent.context): per-section token budgets, stable → volatile order."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from kartrix.agent import context as ctx
from kartrix.context.retrievers.graph import MapEntry
from kartrix.memory.long_term import Recalled


def _memory(i: int, text: str, stale: tuple[str, ...] = ()) -> Recalled:
    return Recalled(str(i), "fact", "project", text, 0.9, datetime.now(UTC), stale)


def test_instructions_keep_the_head_within_budget() -> None:
    text = "Use type hints. " + "x" * 4000
    section = ctx.instructions_section(text, budget=100)
    assert section is not None and section.text.startswith("Use type hints.")
    assert ctx.approx_tokens(section.text) <= 100 and section.text.endswith(ctx.CUT_NOTE.strip())
    report = section.report()
    assert (report.name, report.budget, report.dropped) == ("instructions", 100, 1)
    assert ctx.instructions_section("", 100) is None
    assert ctx.instructions_section("short", 100).dropped == 0  # type: ignore[union-attr]


def test_memories_are_added_whole_best_first_until_the_budget() -> None:
    memories = [_memory(i, f"memory number {i} " + "y" * 60) for i in range(5)]
    section = ctx.memory_section(memories, budget=60)
    assert section is not None
    assert section.items == 2 and section.dropped == 3
    assert "memory number 0" in section.text and "memory number 2" not in section.text
    assert ctx.memory_section([], 60) is None
    stale = ctx.memory_section([_memory(9, "see app/x.py", ("app/x.py",))], 60)
    assert stale is not None and "may be stale" in stale.text


def test_conversation_keeps_the_newest_turns_and_leaves_out_the_request() -> None:
    messages = [
        AIMessage(f"{ctx.SUMMARY_PREFIX}\nUser set up the project."),
        HumanMessage("old question " + "z" * 400),
        AIMessage("old answer " + "z" * 400),
        HumanMessage("recent question"),
        AIMessage("recent answer"),
        HumanMessage("the request"),
    ]
    section = ctx.conversation_section(messages, budget=60)
    assert section is not None
    assert section.text.endswith("User: recent question\nAssistant: recent answer")
    assert "the request" not in section.text and "Summary:" not in section.text
    assert section.items + section.dropped == 5
    whole = ctx.conversation_section(messages, budget=10_000)
    assert whole is not None and whole.text.startswith("Summary: User set up the project.")
    assert ctx.conversation_section([HumanMessage("only the request")], 60) is None


async def test_build_context_orders_sections_and_reports_them(monkeypatch: pytest.MonkeyPatch) -> None:
    async def recall(query: str) -> list[Recalled]:
        assert query == "how do I run the tests"
        return [_memory(1, "tests run with uv run pytest", ("pyproject.toml",))]

    monkeypatch.setattr(ctx.long_term, "recall", recall)
    monkeypatch.setattr(ctx.long_term, "project_instructions", lambda: "Use type hints.")

    async def repo_map() -> list[MapEntry]:
        return [MapEntry("app/db.py", "session", "function", 3, 12)]

    monkeypatch.setattr(ctx.graph, "repo_map", repo_map)
    messages = [HumanMessage("hi"), AIMessage("hello"), HumanMessage("how do I run the tests")]
    text, report = await ctx.build_context(messages)
    titles = ["## Project instructions", "## Repo map", "## Remembered", "## Recent conversation"]
    order = [text.index(t) for t in titles]
    assert order == sorted(order)  # stable → volatile
    assert "app/db.py:3 function session — 12 refs" in text
    assert [s.name for s in report.sections] == ["instructions", "repo_map", "memory", "conversation"]
    assert report.stale == 1 and report.tokens == sum(s.tokens for s in report.sections)


async def test_failed_recall_still_builds_a_context(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(query: str) -> list[Recalled]:
        raise RuntimeError("db down")

    async def no_index() -> list[MapEntry]:
        raise RuntimeError("no index")

    monkeypatch.setattr(ctx.long_term, "recall", broken)
    monkeypatch.setattr(ctx.graph, "repo_map", no_index)
    monkeypatch.setattr(ctx.long_term, "project_instructions", lambda: "")
    text, report = await ctx.build_context([HumanMessage("q")])
    assert text == "" and report.sections == []
