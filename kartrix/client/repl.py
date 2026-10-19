"""The terminal front end over a Kartrix server: the same prompt, rendering, approval prompts and plan
review as the local REPL, but every request runs on the server (``kartrix`` after ``kartrix login``).

The current folder becomes a workspace on the server the first time (a local server only — a hosted
one works on repositories it holds itself). Ctrl+C asks the server to stop the run; a second Ctrl+C
quits. Commands that change local state (``/undo``, ``/mode``, ``/memory`` …) still need ``kartrix --local``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from kartrix.client.api import ApiError, Credentials, KartrixClient
from kartrix.core.events import EVENT_ADAPTER
from kartrix.ui.console import ConsoleRenderer

_HELP = (
    "[dim]/ask <question> · /plan <goal> · /new (new conversation) · /whoami · /exit · Ctrl+C stops a run\n"
    "Other commands (/undo, /mode, /memory, /skills, /mcp …) need the local mode: kartrix --local[/dim]"
)
_LOCAL_ONLY = "isn't available through the server yet — run `kartrix --local` for it"


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


async def find_workspace(client: KartrixClient, folder: Path) -> dict[str, Any]:
    """The server workspace for ``folder``, registered on first use."""
    for ws in await client.workspaces():
        if _same_path(ws["root_path"], str(folder)):
            return ws
    return await client.create_workspace(folder.name or str(folder), str(folder))


class RunView:
    """Shows one run's feed and answers its questions in the terminal."""

    def __init__(self, client: KartrixClient, console: Console) -> None:
        from kartrix.ui.approval_prompt import ConsoleApprover
        from kartrix.ui.plan_review import make_reviewer

        self.client = client
        self.console = console
        self.renderer = ConsoleRenderer(console)
        self.approver = ConsoleApprover(console)
        self.reviewer = make_reviewer(console)

    async def follow(self, run_id: str) -> dict[str, Any]:
        """Render the run until it ends; its result, with ``reported`` True if the core already showed how it ended."""
        result: dict[str, Any] = {}
        reported = False
        async for message in self.client.events(run_id):
            if message.kind == "event":
                try:
                    event = EVENT_ADAPTER.validate_python(message.data)
                except ValidationError:
                    continue  # an event type this client doesn't know yet
                reported = reported or event.type == "run_finished"
                self.renderer(event)
            elif message.kind == "approval_needed":
                await self._approve(run_id, message.data)
            elif message.kind == "plan_review_needed":
                await self._review(run_id, message.data)
            elif message.kind == "run_finished":
                result = message.data
        return {**result, "reported": reported}

    async def _approve(self, run_id: str, data: dict[str, Any]) -> None:
        from kartrix.security.approvals import ApprovalRequest

        requests = [ApprovalRequest(**r) for r in data.get("requests", [])]
        decisions = await self.approver(requests)
        body = [{"type": d.type, "command": d.command, "message": d.message} for d in decisions]
        try:
            await self.client.answer_approval(run_id, data["request_id"], body)
        except ApiError as e:
            if e.status != 404:  # 404: the run was stopped meanwhile
                raise

    async def _review(self, run_id: str, data: dict[str, Any]) -> None:
        from kartrix.tasks.planner import ExecutionPlan

        plan = ExecutionPlan.model_validate(data["plan"])
        review = await self.reviewer(plan)
        approved = review.plan is not None
        edited = review.plan.model_dump(mode="json") if review.plan is not None else None
        try:
            await self.client.answer_plan_review(run_id, data["request_id"], approved, edited, review.feedback)
        except ApiError as e:
            if e.status != 404:
                raise


def _stop_on_ctrl_c(client: KartrixClient, run_id: str, console: Console) -> Any:
    """First Ctrl+C: ask the server to stop the run. Second: quit. Returns a function restoring the handler."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    loop = asyncio.get_running_loop()
    presses = 0

    async def stop() -> None:
        try:
            await client.stop(run_id)
            console.print("[yellow]Stopping… (answer or press Enter at a prompt; it counts as 'no')[/yellow]")
        except ApiError:
            pass  # it finished meanwhile

    def handler(signum: int, frame: Any) -> None:
        nonlocal presses
        presses += 1
        if presses > 1:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        loop.call_soon_threadsafe(lambda: loop.create_task(stop()))

    previous = signal.signal(signal.SIGINT, handler)
    return lambda: signal.signal(signal.SIGINT, previous)


async def run_remote(creds: Credentials, console: Console | None = None) -> int:
    from kartrix.main import in_daemon_thread

    console = console or Console()
    folder = Path.cwd().resolve()
    async with KartrixClient(creds.server, creds.token) as client:
        try:
            me = await client.me()
            workspace = await find_workspace(client, folder)
        except ApiError as e:
            if e.status == 401:
                console.print("[red]Your login has expired — run `kartrix login` again.[/red]")
            elif e.status == 403:
                console.print(
                    f"[red]{escape(creds.server)} doesn't work on folders of your machine. "
                    "Use `kartrix --local` here.[/red]"
                )
            else:
                console.print(f"[red]{escape(e.detail)}[/red]")
            return 1
        except Exception as e:  # httpx.ConnectError and friends
            console.print(f"[red]Can't reach the Kartrix server at {escape(creds.server)}: {escape(str(e))}[/red]")
            console.print("[dim]Start it with `kartrix serve`, or use `kartrix --local`.[/dim]")
            return 1
        console.print("\n[bold blue]Kartrix[/bold blue] — security-first coding agent")
        console.print(
            f"[dim]Server {escape(creds.server)} · {escape(me['email'])} · workspace {escape(workspace['name'])}[/dim]"
        )
        console.print(_HELP + "\n")
        view = RunView(client, console)
        new_session = False
        while True:
            try:
                line = (await in_daemon_thread(console.input, "[bold green]>[/bold green] ")).strip()
            except (EOFError, KeyboardInterrupt):
                line = "/exit"
            if not line:
                continue
            command, _, rest = line.partition(" ")
            if command in ("/exit", "/quit"):
                console.print("[dim]Goodbye![/dim]")
                return 0
            if command == "/help":
                console.print(_HELP)
                continue
            if command == "/whoami":
                console.print(f"{escape(me['email'])} on {escape(creds.server)}")
                continue
            if command == "/new":
                new_session = True
                console.print("[dim]The next request starts a new conversation.[/dim]")
                continue
            if command in ("/ask", "/plan"):
                kind, text = command[1:], rest.strip()
            elif command.startswith("/"):
                console.print(f"[yellow]{escape(command)} {_LOCAL_ONLY}[/yellow]")
                continue
            else:
                kind, text = "ask", line
            if not text:
                console.print(f"[yellow]Usage: /{kind} <text>[/yellow]")
                continue
            try:
                run = await client.start_run(workspace["id"], kind, text, new_session=new_session)
            except ApiError as e:
                console.print(f"[red]{escape(e.detail)}[/red]")
                continue
            new_session = False
            restore = _stop_on_ctrl_c(client, run["id"], console)
            try:
                result = await view.follow(run["id"])
            except ApiError as e:
                console.print(f"[red]{escape(e.detail)}[/red]")
                continue
            except KeyboardInterrupt:
                console.print("[dim]Goodbye! (the run goes on stopping on the server)[/dim]")
                return 130
            finally:
                restore()
            if result.get("reported"):
                continue
            if result.get("status") == "stopped" and result.get("detail"):
                console.print(f"[yellow]Stopped: {escape(str(result['detail']))}[/yellow]")
            elif result.get("status") == "error" and result.get("detail"):
                console.print(f"[red]Error: {escape(str(result['detail']))}[/red]")
