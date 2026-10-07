"""Long-term memory (kartrix.memory.long_term), the per-turn context, conversation compression
(kartrix.agent.context) and the /memory command — with the hashing fake embedder and scripted models."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from kartrix.agent import context as agent_context
from kartrix.agent.tools import remember as remember_tool
from kartrix.config import settings
from kartrix.memory import long_term
from kartrix.security import workspace as ws_mod
from kartrix.ui import commands

pytestmark = pytest.mark.usefixtures("db", "fake_embedder")

GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5" * 4


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.chdir(root)
    ws_mod.set_workspace(root)
    yield root
    ws_mod._current = None


async def test_remember_recall_and_forget(ws: Path) -> None:
    fact = await long_term.remember("tests run with uv run pytest", "fact", "user")
    await long_term.remember("prefers short answers without emojis", "preference", "user")
    recalled = await long_term.recall("how do I run the tests")
    assert recalled[0].id == fact and recalled[0].kind == "fact"
    assert {m.scope for m in await long_term.list_memories()} == {"project", "user"}
    assert await long_term.forget(fact) is True
    assert await long_term.forget(fact) is False and await long_term.forget("not-a-uuid") is False
    assert [m.kind for m in await long_term.list_memories()] == ["preference"]


async def test_near_duplicate_replaces_the_old_memory(ws: Path) -> None:
    first = await long_term.remember("tests run with uv run pytest", "fact", "agent")
    second = await long_term.remember("tests run with uv run pytest -q", "fact", "user")
    rows = await long_term.list_memories()
    assert second == first and len(rows) == 1
    assert rows[0].content == "tests run with uv run pytest -q" and rows[0].source == "user"


async def test_memories_are_per_repository_but_preferences_are_global(ws: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    await long_term.remember("the api lives in app/api/routes", "fact", "user", repo_root=other)
    await long_term.remember("prefers pytest fixtures over setUp", "preference", "user", repo_root=other)
    assert [m.kind for m in await long_term.list_memories()] == ["preference"]


async def test_secrets_are_redacted_and_empty_memories_rejected(ws: Path) -> None:
    await long_term.remember(f"the deploy token is {GITHUB_TOKEN}", "fact", "agent")
    (row,) = await long_term.list_memories()
    assert GITHUB_TOKEN not in row.content
    with pytest.raises(ValueError):
        await long_term.remember("   ", "fact", "user")


async def test_memory_about_a_changed_file_is_marked_stale(ws: Path) -> None:
    (ws / "app").mkdir()
    (ws / "app" / "config.py").write_text("DEBUG = False\n")
    await long_term.remember("settings are read in app/config.py", "fact", "user")
    later = time.time() + 60
    os.utime(ws / "app" / "config.py", (later, later))
    (hit,) = await long_term.recall("where are settings read")
    assert hit.stale_files == ("app/config.py",)
    assert "may be stale" in hit.render()


async def test_context_has_project_instructions_and_recalled_memories(ws: Path) -> None:
    (ws / "KARTRIX.md").write_text("Use type hints everywhere.\n")
    await long_term.remember("tests run with uv run pytest", "fact", "user")
    state = {"messages": [HumanMessage("how do I run the tests?")]}
    context = (await agent_context.assemble(state))["context"]
    assert "Use type hints everywhere." in context
    assert "[fact] tests run with uv run pytest" in context


async def test_remember_tool_stores_an_agent_memory(ws: Path) -> None:
    out = await remember_tool.ainvoke({"content": "prefers pathlib over os.path", "kind": "preference"})
    assert out.startswith("Saved preference")
    (row,) = await long_term.list_memories()
    assert (row.scope, row.source) == ("user", "agent")


async def test_memory_command_add_list_forget(ws: Path) -> None:
    await commands.handle_memory_command("add --user answer briefly", "s1")
    await commands.handle_memory_command("add migrations live in migrations/versions", "s1")
    rows = await long_term.list_memories()
    assert sorted((m.kind, m.scope) for m in rows) == [("fact", "project"), ("preference", "user")]
    await commands.handle_memory_command("", "s1")
    await commands.handle_memory_command(f"forget {str(rows[0].id)[:8]}", "s1")
    assert len(await long_term.list_memories()) == 1


async def test_compress_summarises_old_turns(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.memory, "summarize_at_tokens", 50)
    monkeypatch.setattr(settings.memory, "keep_last_messages", 2)
    prompts: list[Any] = []

    class Model(GenericFakeChatModel):
        async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            prompts.append(messages)
            return await super().ainvoke(messages, *args, **kwargs)

    model = Model(messages=iter([AIMessage("User is fixing add() in app.py.")]))
    monkeypatch.setattr(agent_context, "get_chat_model", lambda role="main", **kw: model)
    history = add_messages(
        [],
        [
            HumanMessage("what does add do? " * 20),
            AIMessage("it subtracts " * 20),
            HumanMessage("fix it"),
            AIMessage("fixed"),
            HumanMessage("now add a test"),
        ],
    )
    update = await agent_context.compress({"messages": history})
    result = add_messages(history, update["messages"])
    assert [m.content for m in result][1:] == ["fixed", "now add a test"]
    assert result[0].content.startswith(agent_context.SUMMARY_PREFIX)
    assert "fixing add()" in result[0].content
    assert "what does add do?" in prompts[0][0].content


async def test_short_conversation_is_not_compressed(ws: Path) -> None:
    assert await agent_context.compress({"messages": [HumanMessage("hi")]}) == {}
