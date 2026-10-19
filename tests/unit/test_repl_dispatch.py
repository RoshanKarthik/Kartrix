"""The local REPL's dispatch: plain text is a request (like /ask); unknown /commands show the help."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from kartrix.ui import commands


class FakeCore:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.session_id = "s"

    async def ask(self, question: str, limits: Any = None) -> None:
        self.asked.append(question)


async def test_plain_text_and_ask_go_to_the_agent() -> None:
    core = FakeCore()
    await commands.dispatch(core, "what does slugify do?")  # type: ignore[arg-type]
    await commands.dispatch(core, "/ask where is it tested?")  # type: ignore[arg-type]
    assert core.asked == ["what does slugify do?", "where is it tested?"]


async def test_unknown_commands_show_help(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=200, color_system=None)
    monkeypatch.setattr(commands, "console", console)
    core = FakeCore()
    await commands.dispatch(core, "/nope")  # type: ignore[arg-type]
    assert core.asked == [] and "Unknown command" in console.export_text()
