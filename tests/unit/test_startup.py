"""Fast startup (2.2): the prompt path stays light, the REPL answers before the core is up, background
work never prints over the prompt, the index state reaches search results, fallback models are lazy."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import pytest
from rich.console import Console

from kartrix.context import index_status
from kartrix.core import events as ev

# Modules that take seconds to import on a cold start; none may load before the prompt shows.
HEAVY = ("langchain", "langchain_core", "langgraph", "sqlalchemy", "asyncpg", "tree_sitter", "openai",
         "langchain_openai", "langchain_nvidia_ai_endpoints", "redis", "redisvl", "watchdog", "kartrix.core.session")  # fmt: skip


def test_prompt_path_imports_nothing_heavy() -> None:
    code = f"import sys, kartrix.cli, kartrix.main; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == "", f"imported before the prompt: {out.stdout.strip()}"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from kartrix.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0 and capsys.readouterr().out.startswith("kartrix ")


async def test_in_daemon_thread_returns_and_raises() -> None:
    from kartrix.main import in_daemon_thread

    assert await in_daemon_thread(lambda a, b: a + b, 2, 3) == 5

    def boom() -> None:
        raise EOFError

    with pytest.raises(EOFError):
        await in_daemon_thread(boom)


def test_renderer_holds_events_while_the_prompt_waits() -> None:
    from kartrix.ui.console import ConsoleRenderer

    console = Console(record=True, width=120, color_system=None)
    renderer = ConsoleRenderer(console)
    renderer.hold()
    renderer(ev.Notice(text="index ready"))
    assert console.export_text(clear=False) == ""  # nothing printed over the user's typing
    renderer.release()
    renderer(ev.Notice(text="live again"))
    assert console.export_text() == "index ready\nlive again\n"


async def test_repl_answers_help_and_exit_before_the_core_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    import kartrix.main as repl

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def never_ready() -> Any:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    inputs = iter(["/help", "/exit"])
    monkeypatch.setattr(repl, "_start_core", never_ready)
    monkeypatch.setattr(repl, "_ask", lambda: next(inputs))
    console = Console(record=True, width=200, color_system=None)
    monkeypatch.setattr(repl, "console", console)
    await asyncio.wait_for(repl._run_async(), 10)
    out = console.export_text()
    assert "/ask <question>" in out and "Goodbye!" in out
    assert started.is_set() and cancelled.is_set()  # quitting early cancels the startup


async def test_repl_reports_a_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import kartrix.main as repl

    async def broken() -> Any:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    monkeypatch.setattr(repl, "_start_core", broken)
    monkeypatch.setattr(repl, "_ask", lambda: "/ask hi")
    console = Console(record=True, width=200, color_system=None)
    monkeypatch.setattr(repl, "console", console)
    await asyncio.wait_for(repl._run_async(), 10)
    assert "Kartrix could not start: NVIDIA_API_KEY is not set" in console.export_text()


async def test_search_says_when_the_index_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.agent import tools

    async def fake_retrieve(query: str) -> list[dict[str, Any]]:
        return [{"source": "a.py", "start_line": 1, "end_line": 2, "type": "function", "name": "f", "content": "x"}]

    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    try:
        index_status.set_status("building")
        assert "still being built" in await tools.search_codebase.ainvoke({"query": "f"})
        index_status.set_status("failed", "embedder down")
        assert "embedder down" in await tools.search_codebase.ainvoke({"query": "f"})
        index_status.set_status("ready")
        assert "Note:" not in await tools.search_codebase.ainvoke({"query": "f"})
    finally:
        index_status.set_status("unknown")


def test_fallback_models_are_built_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.llm import factory

    built: list[str] = []
    monkeypatch.setattr(factory, "_build_chat_model", lambda provider, model, **kw: built.append(model) or model)
    middleware = factory.get_model_middleware()
    lazy = next(m for m in middleware if isinstance(m, factory.LazyModelFallbackMiddleware))
    assert built == []  # nothing imported or built at startup
    assert lazy.models == [fb.model for fb in factory.settings.llm.fallbacks]
    assert lazy.models is lazy.models and len(built) == len(factory.settings.llm.fallbacks)  # built once


def test_missing_fallback_key_still_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.llm import factory

    if not any(fb.provider == "huggingface" for fb in factory.settings.llm.fallbacks):
        pytest.skip("no Hugging Face fallback configured")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(factory.ProviderConfigError, match="HF_TOKEN"):
        factory.get_model_middleware()
