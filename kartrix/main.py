"""The interactive REPL: one front end of :mod:`kartrix.core` (renders its events, asks the user).

Fast start: the prompt appears at once. The heavy part — importing the agent stack and
:meth:`CoreSession.start` (model clients, sandbox, database, agent) — runs in the background,
and the code index after it. Typing never waits; a command that needs the core waits only for
what is still starting. ``/help`` and ``/exit`` always answer immediately. Messages from the
background are held while the prompt waits for input and shown with the next command.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import threading
import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

from kartrix.core.events import bus
from kartrix.observability.logger import get_logger
from kartrix.ui.console import ConsoleRenderer

console = Console()
logger = get_logger(__name__)

_HELP = (
    "[dim]/ask <question> · /plan <goal> · /undo · /mode · /help for all commands · /exit to quit · "
    "Ctrl+C stops a run[/dim]"
)


def in_daemon_thread[T](fn: Callable[..., T], *args: Any) -> asyncio.Future[T]:
    """Run a blocking call (reading the terminal, slow imports) without tying up the event loop —
    in a daemon thread, so an unanswered prompt never keeps the process alive at exit."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def deliver(result: Any = None, error: BaseException | None = None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def target() -> None:
        try:
            result = fn(*args)
        except BaseException as e:
            loop.call_soon_threadsafe(deliver, None, e)
        else:
            loop.call_soon_threadsafe(deliver, result)

    threading.Thread(target=target, name=f"kartrix-{getattr(fn, '__name__', 'call')}", daemon=True).start()
    return future


def _import_core() -> None:
    for name in ("kartrix.core.session", "kartrix.ui.commands", "kartrix.ui.approval_prompt", "kartrix.ui.plan_review"):
        importlib.import_module(name)


async def _start_core() -> Any:
    started = time.perf_counter()
    await in_daemon_thread(_import_core)
    imported = time.perf_counter()
    from kartrix.core.session import CoreSession
    from kartrix.ui.approval_prompt import ConsoleApprover
    from kartrix.ui.plan_review import make_reviewer

    core = await CoreSession.start(ConsoleApprover(console), make_reviewer(console))
    logger.info(
        "Core ready",
        extra={"imports_s": round(imported - started, 2), "start_s": round(time.perf_counter() - imported, 2)},
    )
    return core


def _ask() -> str:
    return Prompt.ask("[bold green]>[/bold green]", console=console)


async def _wait_for(starter: asyncio.Task[Any]) -> Any:
    """The started core; tells the user when they are waiting for it."""
    if not starter.done():
        console.print("[dim]Still starting (model clients, sandbox, database)…[/dim]")
    return await starter


async def _run_async() -> None:
    t0 = time.perf_counter()
    logger.info("Starting Kartrix")
    renderer = ConsoleRenderer(console)
    unsubscribe = bus.subscribe(renderer)
    starter = asyncio.create_task(_start_core(), name="kartrix-start")
    console.print("\n[bold blue]Kartrix[/bold blue] — security-first coding agent")
    console.print(_HELP + "\n")
    logger.info("Prompt ready", extra={"seconds": round(time.perf_counter() - t0, 3)})
    core: Any = None
    try:
        while True:
            renderer.hold()  # nothing prints over what the user is typing
            try:
                user_input = await in_daemon_thread(_ask)
            except EOFError:
                user_input = "/exit"
            finally:
                renderer.release()
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit"):
                logger.info("Shutting down")
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input == "/help":
                if starter.done() and not starter.exception():
                    from kartrix.ui.commands import print_help

                    print_help()
                else:
                    console.print(_HELP)
                continue
            try:
                core = await _wait_for(starter)
            except Exception as e:
                console.print(f"[red]Kartrix could not start: {escape(str(e))}[/red]")
                break
            from kartrix.ui.commands import dispatch

            await dispatch(core, user_input)
    finally:
        if core is None and starter.done() and not starter.cancelled() and starter.exception() is None:
            core = starter.result()  # started, but no command used it yet
        if core is not None:
            await core.close()
        elif not starter.done():
            starter.cancel()  # quitting before startup finished
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await starter
        unsubscribe()


def run() -> None:
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
