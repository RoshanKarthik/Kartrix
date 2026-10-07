"""Terminal prompt that answers approval requests (the CLI :data:`~kartrix.security.approvals.Approver`).

The command is shown exactly as it will run: control characters (ANSI escapes could hide or
rewrite text on screen), invisible and bidi-override characters are printed as visible
``\\xNN`` / ``\\uNNNN`` escapes, and Rich markup in it is not interpreted.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from kartrix.security.approvals import ApprovalDecision, ApprovalRequest

_HIDDEN = re.compile(r"[\x00-\x1f\x7f-\x9f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]")

AskFn = Callable[[str, list[str] | None, str], str]


def visible(text: str) -> str:
    """Make control/invisible characters visible so the user sees what really runs."""

    def esc(m: re.Match[str]) -> str:
        code = ord(m.group())
        return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"

    return _HIDDEN.sub(esc, text)


def _rich_ask(console: Console) -> AskFn:
    def ask(prompt: str, choices: list[str] | None, default: str) -> str:
        return Prompt.ask(prompt, choices=choices, default=default, console=console, show_default=bool(default))

    return ask


class ConsoleApprover:
    """Asks in the terminal, one request at a time (a lock keeps concurrent tasks from interleaving).
    Without an interactive terminal every request is declined, with a reason the agent sees."""

    def __init__(self, console: Console | None = None, ask: AskFn | None = None, interactive: bool | None = None):
        self.console = console or Console()
        self._ask = ask or _rich_ask(self.console)
        self._interactive = interactive
        self._lock = asyncio.Lock()

    async def __call__(self, requests: list[ApprovalRequest]) -> list[ApprovalDecision]:
        interactive = self._interactive if self._interactive is not None else sys.stdin.isatty()
        if not interactive:
            reason = "no interactive terminal to ask the user"
            return [ApprovalDecision("reject", message=reason) for _ in requests]
        async with self._lock:
            decisions: list[ApprovalDecision] = []
            for i, request in enumerate(requests, 1):
                try:
                    decisions.append(await self._decide(request, i, len(requests)))
                except EOFError:
                    decisions.append(ApprovalDecision("reject", message="no answer (input closed)"))
            return decisions

    def _show(self, request: ApprovalRequest, index: int, total: int) -> None:
        body = Text()
        body.append("$ ", style="dim")
        body.append(visible(request.command), style="bold")
        where = f"in {visible(request.directory)}"
        if request.task_key:
            where += f" · task {visible(request.task_key)}"
        body.append(f"\n{where} · mode {request.mode}", style="dim")
        body.append(f"\n{visible(request.reason)}")
        if request.alongside:
            body.append("\nAlso in this step: " + ", ".join(visible(a) for a in request.alongside), style="dim")
        counter = f" ({index}/{total})" if total > 1 else ""
        title = Text(f"Approval needed{counter} · {request.category}", style="bold yellow")
        self.console.print(Panel(body, title=title, title_align="left", border_style="yellow", expand=False))

    async def _decide(self, request: ApprovalRequest, index: int, total: int) -> ApprovalDecision:
        self._show(request, index, total)
        options = ["y", "a", "e", "n"] if request.allow_session else ["y", "e", "n"]
        session = "[a] this exact command for the session · " if request.allow_session else ""
        self.console.print(Text(f"[y] yes, once · {session}[e] edit · [n] no", style="dim"))
        while True:
            choice = await asyncio.to_thread(self._ask, "Run it?", options, "n")
            if choice == "y":
                return ApprovalDecision("approve")
            if choice == "a" and request.allow_session:
                return ApprovalDecision("approve_session")
            if choice == "n":
                reason = (await asyncio.to_thread(self._ask, "Reason for the agent (optional)", None, "")).strip()
                return ApprovalDecision("reject", message=reason or None)
            if choice == "e":
                edited = (await asyncio.to_thread(self._ask, "Command to run instead (empty = back)", None, "")).strip()
                if edited:
                    return ApprovalDecision("edit", command=edited)
