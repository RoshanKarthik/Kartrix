"""The terminal approval prompt: answers, edits, non-interactive use and safe display."""

from __future__ import annotations

import io

from rich.console import Console

from kartrix.security.approvals import ApprovalDecision, ApprovalRequest
from kartrix.security.injection import visible
from kartrix.ui.approval_prompt import ConsoleApprover


def request(command: str = "npm install express", allow_session: bool = True, **kw: object) -> ApprovalRequest:
    fields: dict = {
        "tool_call_id": "call_0",
        "tool": "run_command",
        "command": command,
        "directory": ".",
        "category": "install",
        "reason": "installs packages",
        "mode": "default",
        "allow_session": allow_session,
    }
    return ApprovalRequest(**{**fields, **kw})


def approver(*answers: str) -> tuple[ConsoleApprover, list[tuple[str, list[str] | None]], io.StringIO]:
    out = io.StringIO()
    queue = list(answers)
    prompts: list[tuple[str, list[str] | None]] = []

    def ask(prompt: str, choices: list[str] | None, default: str) -> str:
        prompts.append((prompt, choices))
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = Console(file=out, width=200, color_system=None)
    return ConsoleApprover(console, ask=ask, interactive=True), prompts, out


async def test_each_answer_maps_to_a_decision() -> None:
    a, prompts, _ = approver("y", "a", "n", "too risky", "n", "")
    decisions = await a([request(), request(), request(), request()])
    assert decisions == [
        ApprovalDecision("approve"),
        ApprovalDecision("approve_session"),
        ApprovalDecision("reject", message="too risky"),
        ApprovalDecision("reject"),
    ]
    assert prompts[0] == ("Run it?", ["y", "a", "e", "n"])


async def test_edit_and_back() -> None:
    a, _, _ = approver("e", "", "e", "npm install express@5")
    assert await a([request()]) == [ApprovalDecision("edit", command="npm install express@5")]


async def test_session_option_hidden_for_destructive_commands() -> None:
    a, prompts, out = approver("y")
    await a([request("git clean -fdx", allow_session=False)])
    assert prompts[0][1] == ["y", "e", "n"] and "for the session" not in out.getvalue()


async def test_closed_input_declines_the_rest() -> None:
    a, _, _ = approver("y")
    decisions = await a([request(), request()])
    assert decisions == [ApprovalDecision("approve"), ApprovalDecision("reject", message="no answer (input closed)")]


async def test_without_a_terminal_everything_is_declined() -> None:
    a = ConsoleApprover(Console(file=io.StringIO()), ask=lambda *_: "y", interactive=False)
    decisions = await a([request(), request()])
    assert [d.type for d in decisions] == ["reject", "reject"]
    assert decisions[0].message == "no interactive terminal to ask the user"


async def test_display_shows_the_real_command() -> None:
    sneaky = "npm install ok\x1b[2K\rpkg \u202eexe.cmd [bold]x[/bold]"
    a, _, out = approver("n", "")
    await a([request(sneaky, task_key="task_4", alongside=["write_file app.py"])])
    text = out.getvalue()
    assert "\x1b" not in text and "\u202e" not in text
    assert r"\x1b[2K\x0dpkg \u202eexe.cmd [bold]x[/bold]" in text  # markup is not interpreted either
    assert "task task_4" in text and "Also in this step: write_file app.py" in text


def test_visible() -> None:
    assert visible("ls -la") == "ls -la"
    assert visible("a\tb\x00c\u200bd\ufeff") == "a\\x09b\\x00c\\u200bd\\ufeff"
    assert visible("naïve café") == "naïve café"  # ordinary non-ASCII stays readable


async def test_mcp_requests_cannot_be_edited_or_session_allowed() -> None:
    a, prompts, out = approver("e", "y")  # "e" is not an option here: asked again
    req = request('github: create_issue {"title": "x"}', allow_session=False, directory="", editable=False)
    assert await a([req]) == [ApprovalDecision("approve")]
    assert prompts[0][1] == ["y", "n"]
    text = out.getvalue()
    assert "[e] edit" not in text and "$ github" not in text and "in ." not in text


async def test_a_stopped_run_declines_without_asking_further() -> None:
    from kartrix.config import BudgetLimits
    from kartrix.security.budget import Budget, budget_scope

    budget = Budget("turn", BudgetLimits())
    ap, asked, _ = approver("y", "y")
    with budget_scope(budget):
        budget.stop("stopped by the user (Ctrl+C)", hard=True)  # e.g. Ctrl+C while the prompt was open
        decisions = await ap([request(), request("npm test")])
    assert asked == []
    assert [d.type for d in decisions] == ["reject", "reject"]
    assert "run stopped" in (decisions[0].message or "")
