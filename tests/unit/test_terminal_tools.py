"""run_command end to end: policy → real process (no shell), env scrubbing, timeouts, output."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kartrix.config import settings
from kartrix.sandbox.base import SandboxError, SandboxRun
from kartrix.sandbox.manager import set_sandbox
from kartrix.security import permissions
from kartrix.security import workspace as ws_mod
from kartrix.security.command_policy import evaluate
from kartrix.security.workspace import set_workspace
from kartrix.tools.filesystem_tools import write_file
from kartrix.tools.process_runner import Launch
from kartrix.tools.terminal_tools import run_command
from tests.fakes import FakeSandbox


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    previous = ws_mod._current
    set_workspace(root)
    yield root
    ws_mod._current = previous
    permissions._mode = None
    permissions.clear_session_allowances()


def run(command: str, directory: str = ".") -> str:
    return run_command.invoke({"command": command, "directory": directory})


def script(root: Path, name: str, code: str) -> str:
    (root / name).write_text(code)
    return name


@pytest.fixture
def auto(root: Path) -> Path:
    permissions.set_mode("auto")
    return root


def test_runs_without_shell_in_directory(auto: Path) -> None:
    (auto / "sub").mkdir()
    script(auto / "sub", "where.py", "import os; print(os.getcwd())")
    assert run("python where.py", "sub").strip().endswith("sub")


def test_secrets_not_inherited(auto: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-should-not-leak")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    script(auto, "env.py", "import os; print(os.environ.get('NVIDIA_API_KEY'), os.environ.get('HARMLESS_SETTING'))")
    assert run("python env.py").strip() == "None visible"


def test_exit_code_stderr_and_no_stdin(auto: Path) -> None:
    script(auto, "fail.py", "import sys; print('out'); print('bad', file=sys.stderr); sys.exit(3)")
    assert run("python fail.py") == "out\nSTDERR: bad\nExit code: 3"
    script(auto, "ask.py", "print(repr(input()))")  # stdin is closed → EOFError instead of hanging
    assert "EOFError" in run("python ask.py")


def test_timeout_kills_process_tree(auto: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.permissions, "command_timeout", 1.5)
    script(
        auto,
        "spawn.py",
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n",
    )
    start = time.monotonic()
    out = run("python spawn.py")
    assert "timed out after 2 seconds" in out or "timed out after 1 seconds" in out
    assert time.monotonic() - start < 20  # communicate() returned: the grandchild didn't keep the pipes open


def test_long_output_is_clipped(auto: Path) -> None:
    script(auto, "loud.py", "print('x' * 100_000)")
    out = run("python loud.py")
    assert "characters omitted" in out and len(out) < 35_000


def test_denied_and_ask_messages(root: Path) -> None:
    set_sandbox(None)  # without a sandbox, running project code needs approval in default mode
    assert run("sudo ls").startswith("Error: command denied")
    assert run("ls | head").startswith("Error: command denied")
    script(root, "x.py", "print('ran')")
    out = run("python x.py")  # RUN needs approval in default mode; no handler yet → not run
    assert "needs the user's approval" in out and "/mode auto" in out


def test_user_approval_and_session_allowance(root: Path) -> None:
    set_sandbox(None)
    script(root, "x.py", "print('ran')")
    with permissions.user_approved():  # what the approval middleware sets for an approved call
        assert run("python x.py").strip() == "ran"
    assert "needs the user's approval" in run("python x.py")  # only that call
    with permissions.user_approved():
        assert run("sudo ls").startswith("Error: command denied")  # approval never overrides deny

    permissions.allow_for_session(evaluate("python x.py"))
    assert run("python x.py").strip() == "ran"
    assert "needs the user's approval" in run("python x.py -v")  # exact command only
    (root / "sub").mkdir()
    assert "needs the user's approval" in run("python ../x.py", "sub")  # same script, other directory
    permissions.clear_session_allowances()
    assert "needs the user's approval" in run("python x.py")


def test_read_only_mode_blocks_file_changes(root: Path) -> None:
    permissions.set_mode("read_only")
    assert "read-only mode" in write_file.invoke({"file_path": "a.txt", "content": "x"})
    assert not (root / "a.txt").exists()
    assert run(f'"{Path(sys.executable).name}" --version').startswith("Python")  # still allowed: read


def test_background_leftovers_are_killed(auto: Path) -> None:
    # The command exits at once but leaves a child that would write a file 2 s later.
    script(
        auto,
        "detach.py",
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time, pathlib; time.sleep(2); "
        'pathlib.Path("leftover.txt").write_text("x")\'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n',
    )
    assert run("python detach.py") == "(no output)"
    time.sleep(3)
    assert not (auto / "leftover.txt").exists()


def test_commands_run_through_the_sandbox(root: Path, fake_sandbox: FakeSandbox) -> None:
    script(root, "x.py", "print('ran')")
    assert run("python x.py").strip() == "ran"  # default mode: sandboxed project code runs without asking
    (sandboxed,) = fake_sandbox.runs
    assert sandboxed.network == "off" and sandboxed.workspace == root and sandboxed.argv == ["python", "x.py"]
    assert not any("KEY" in k for k in sandboxed.env)  # secrets were removed before the sandbox saw it
    assert run("git --version")  # git runs outside the sandbox
    assert len(fake_sandbox.runs) == 1


def test_sandbox_setup_failure_is_reported(root: Path) -> None:
    class Broken(FakeSandbox):
        def prepare(self, run: SandboxRun) -> Launch:
            raise SandboxError("could not set up the sandbox's file access: boom")

    set_sandbox(Broken())
    script(root, "x.py", "print('ran')")
    assert run("python x.py") == "Error: command not run — could not set up the sandbox's file access: boom"


def test_review_command_runs_checks_but_never_downloads_installs_or_writes(root: Path) -> None:
    """The reviewer verifies the change as it is: a download it wanted once turned into an approval request
    and a review round the coder could never satisfy."""
    from kartrix.security.approvals import ApprovalMiddleware
    from kartrix.tools.terminal_tools import REVIEW_REFUSED, review_command

    permissions.set_mode("auto")  # even where these would run without asking
    script(root, "check.py", "print('checked')")

    def review(command: str) -> str:
        return review_command.invoke({"command": command})

    assert review("python check.py").strip() == "checked"
    for command in ("curl -o x.txt https://example.com/x.txt", "pip install requests", "git commit -m x"):
        assert review(command).startswith("Error: not run — a review only checks the change"), command
    assert review_command.name == "run_command"  # same tool name in the prompts and the approval middleware

    permissions.set_mode("default")
    call = {"name": "run_command", "args": {"command": "curl -o x.txt https://example.com/x.txt"}, "id": "c1"}
    assert ApprovalMiddleware()._request(call, [call]) is not None  # the coder is asked about it
    assert ApprovalMiddleware(refuse=REVIEW_REFUSED)._request(call, [call]) is None  # the reviewer never is
