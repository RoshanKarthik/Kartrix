"""The agent's command tool. Every command passes the command policy first
(``kartrix.security.command_policy``) and then runs without a shell, inside the workspace,
with Kartrix's secrets removed from its environment and stdin closed.
"""

from __future__ import annotations

from langchain.tools import tool

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.security.audit import note
from kartrix.security.command_policy import Decision, evaluate
from kartrix.security.environment import scrubbed_env
from kartrix.security.permissions import get_mode, has_approval_handler, request_approval
from kartrix.tools.process_runner import ProcessResult, run_process

logger = get_logger(__name__)

_OUTPUT_HEAD_CHARS = 10_000
_OUTPUT_TAIL_CHARS = 20_000  # errors usually sit at the end


def _clip(text: str) -> str:
    if len(text) <= _OUTPUT_HEAD_CHARS + _OUTPUT_TAIL_CHARS:
        return text
    skipped = len(text) - _OUTPUT_HEAD_CHARS - _OUTPUT_TAIL_CHARS
    return f"{text[:_OUTPUT_HEAD_CHARS]}\n… [{skipped} characters omitted] …\n{text[-_OUTPUT_TAIL_CHARS:]}"


def format_result(result: ProcessResult, timeout: float) -> str:
    parts = []
    if result.stdout.strip():
        parts.append(_clip(result.stdout.rstrip()))
    if result.stderr.strip():
        parts.append(f"STDERR: {_clip(result.stderr.rstrip())}")
    if result.timed_out:
        parts.append(f"Error: command timed out after {timeout:.0f} seconds and was stopped")
    elif result.returncode != 0:
        parts.append(f"Exit code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"


def _not_approved(decision: Decision) -> str:
    why = f"{decision.category}: {decision.reason}"
    if has_approval_handler():
        return f"Error: the user declined this command ({why})"
    mode = get_mode()
    switch = "" if mode == "auto" else ", switch mode with /mode auto"
    return (
        f"Error: command not run — it needs the user's approval ({why}) and permission mode "
        f"'{mode}' doesn't allow it automatically. Tell the user the exact command; they can run it "
        f"themselves{switch}, or add a rule under permissions.allow in config.yaml."
    )


@tool(parse_docstring=True)
def run_command(command: str, directory: str = ".") -> str:
    """Run one program with its arguments, e.g. "pytest -q" or "npm run build". There is no
    shell: pipes, &&, ;, redirects and $(...) are not supported, and cmd/bash built-ins aren't
    available — use the file tools for reading, searching and editing files. stdin is closed,
    so pass flags that avoid interactive prompts (e.g. "npm init -y").

    Args:
        command: The program and its arguments.
        directory: Directory to run in, relative to the workspace root.
    """
    decision = evaluate(command, directory)
    note(policy=decision.action, category=str(decision.category), reason=decision.reason, mode=get_mode())
    if decision.action == "deny":
        return f"Error: command denied — {decision.reason}"
    if decision.action == "ask" and not request_approval(decision):
        return _not_approved(decision)

    timeout = settings.permissions.command_timeout
    try:
        result = run_process(decision.run_args, decision.cwd, scrubbed_env(), timeout)
    except OSError as e:
        return f"Error: could not start {decision.argv[0]}: {e.strerror or e}"
    note(returncode=result.returncode, timed_out=result.timed_out)
    logger.info(
        "Command finished",
        extra={"argv": decision.argv, "returncode": result.returncode, "timed_out": result.timed_out},
    )
    return format_result(result, timeout)
