from __future__ import annotations

import json
import uuid
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from kartrix.agent.factory import agent_middleware
from kartrix.agent.graph import WORKING_RULES
from kartrix.agent.tools import search_codebase, symbol_graph
from kartrix.llm.factory import get_chat_model, get_model_middleware
from kartrix.observability.logger import get_logger
from kartrix.security.approvals import (
    ApprovalDecision,
    ApprovalMiddleware,
    ApprovalRequest,
    Approver,
    stream_agent,
)
from kartrix.security.budget import BudgetMiddleware, raise_if_stopped
from kartrix.security.injection import SECURITY_RULES
from kartrix.tools.filesystem_tools import READ_TOOLS, WRITE_TOOLS, edit_file, write_file
from kartrix.tools.terminal_tools import run_command

logger = get_logger(__name__)


# Each task type gets a minimal, focused toolset — least privilege per task.
# All file tools are jailed to the workspace (kartrix.security.workspace).
# Every task can search the code; tasks that change behaviour can run the tests (commands go through the
# policy and the sandbox like everywhere else).
_SEARCH = [search_codebase, symbol_graph]
_TOOLS_BY_TYPE: dict[str, list] = {
    "design": [*_SEARCH, *READ_TOOLS, write_file, edit_file],
    "implement": [*_SEARCH, *READ_TOOLS, *WRITE_TOOLS, run_command],
    "test": [*_SEARCH, *READ_TOOLS, *WRITE_TOOLS, run_command],
    "review": [*_SEARCH, *READ_TOOLS, write_file, edit_file, run_command],
    "integrate": [*_SEARCH, *READ_TOOLS, *WRITE_TOOLS, run_command],
    "configure": [*_SEARCH, *READ_TOOLS, *WRITE_TOOLS, run_command],
}

_DEFAULT_TOOLS = [*_SEARCH, *READ_TOOLS, write_file, edit_file]


def _parse_json_field(val) -> list:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val or []


def _build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
    """
    Build a task-specific system prompt.

    dep_outputs — results from completed dependency tasks — are injected as
    "PRIOR TASK OUTPUTS" so the agent has memory of what was already built,
    without needing a shared checkpointer across tasks.
    """
    criteria_lines = "\n".join(f"  - {c}" for c in _parse_json_field(task.get("acceptance_criteria")))
    output_files = "\n".join(f"  - {f}" for f in _parse_json_field(task.get("output_files")))

    prior_context = ""
    if dep_outputs:
        parts = [f"[{dep['id']}] {dep['title']}\n{dep['result'] or '(no output recorded)'}" for dep in dep_outputs]
        prior_context = "\n\nPRIOR TASK OUTPUTS (from your dependencies):\n" + "\n\n".join(parts)

    return f"""You are an expert software engineer executing one task of an approved plan in this repository.
Work in the existing code: find what you need with search_codebase/read_file, read a file before changing it,
use edit_file for changes to existing files and write_file only for new files. Match the existing style.
When the task changes behaviour, run the project's tests with run_command and fix what fails.

{WORKING_RULES}

TASK ID:   {task["id"]}
TASK TYPE: {task["task_type"]}
TITLE:     {task["title"]}

DESCRIPTION:
{task["description"]}

FILES TO PRODUCE:
{output_files or "  (none specified)"}

ACCEPTANCE CRITERIA (your output must satisfy ALL of these):
{criteria_lines or "  (none specified)"}{prior_context}

When done, summarise in 3-5 bullet points: the files you changed, what you changed and the test result.
Don't leave the implementation incomplete; if something could not be done, say so plainly.

{SECURITY_RULES}"""


# ------------------------------------------------------------------
# LLM-as-judge
# ------------------------------------------------------------------


class _JudgeVerdict(BaseModel):
    passed: bool
    score: int  # 0-10
    reason: str  # one sentence


_JUDGE_SYSTEM_PROMPT = """\
You are a code review judge. Given a task description, its acceptance criteria,
and the AI agent's output, decide whether the task was completed satisfactorily.

Score 0-10:
  8-10 → passed (all acceptance criteria met)
  5-7  → borderline (minor gaps, still usable)
  0-4  → failed (criteria not met, re-execution needed)

Set passed=true if score >= 6. One short sentence for reason."""


async def _judge_task(task: dict, agent_output: str) -> _JudgeVerdict:
    """
    Lightweight LLM-as-judge. Uses judge_model (cheaper) from config.
    Returns a structured verdict with passed/score/reason.
    """
    llm = get_chat_model("judge", temperature=0)
    judge_agent = create_agent(
        llm,
        tools=[],
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        response_format=_JudgeVerdict,
        middleware=[*get_model_middleware(temperature=0), BudgetMiddleware()],
    )

    criteria = _parse_json_field(task.get("acceptance_criteria"))
    user_message = f"""TASK: {task["title"]}
DESCRIPTION: {task["description"]}
ACCEPTANCE CRITERIA: {json.dumps(criteria, indent=2)}

AGENT OUTPUT:
{agent_output[:2000]}"""

    result = await judge_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    raise_if_stopped()  # the budget ended the judge before it answered
    return result["structured_response"]


async def _no_approver(requests: list[ApprovalRequest]) -> list[ApprovalDecision]:
    raise RuntimeError("approval requested without an approver")  # unreachable: no ApprovalMiddleware then


# ------------------------------------------------------------------
# Main entry point — called by orchestrator._execute()
# ------------------------------------------------------------------


async def run_subtask_agent(task: dict, dep_outputs: list[dict] | None = None, approver: Approver | None = None) -> str:
    """
    Build a fresh agent for a single task and invoke it.

    After the agent returns, an LLM judge verifies the output against
    acceptance_criteria. If it fails (score < 6), raises ValueError so
    the orchestrator's existing retry logic kicks in automatically.

    With an ``approver``, commands that need approval pause the agent until it answers; without
    one (no one to ask) they are refused and the agent is told why.
    """
    llm = get_chat_model("main", temperature=0)

    tools = _TOOLS_BY_TYPE.get(task.get("task_type", ""), _DEFAULT_TOOLS)
    system_prompt = _build_system_prompt(task, dep_outputs or [])

    logger.info(f"Building agent for task {task['id']} (type={task['task_type']}, tools={[t.name for t in tools]})")

    # Without an approver there's no one to ask: commands needing approval are refused by the tool.
    middleware = agent_middleware(ApprovalMiddleware() if approver is not None else None, temperature=0)
    # Interrupts need a checkpointer; a task's run is not resumed across restarts (recovery
    # re-runs crashed tasks), so memory is enough. The thread id is not a UUID on purpose:
    # audit rows then take the session id from the orchestrator's audit scope.
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": f"task-{task['id']}-{uuid.uuid4().hex[:8]}"}}

    # Existing repositories are the normal case: never overwrite code the agent hasn't read. A new project
    # (empty folder) follows the same rules — there is just nothing to read first.
    user_message = (
        f"{task['description']}\n\n"
        "Do this task in the current repository. If a file you need to change exists, read it first and edit it; "
        "create only the files this task needs."
    )

    final_state: dict[str, Any] = {"messages": []}  # stays empty if the stream yields nothing
    async for step in stream_agent(
        agent, {"messages": [{"role": "user", "content": user_message}]}, config, approver or _no_approver
    ):
        last_msg = step["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            logger.info(f"Task {task['id']} → tool calls: {[tc['name'] for tc in tool_calls]}")
        else:
            logger.info(f"Task {task['id']} → {type(last_msg).__name__}: {str(getattr(last_msg, 'content', ''))[:200]}")
        final_state = step
    # A budget limit or the kill switch ended the agent early: the task isn't done (and isn't judged).
    raise_if_stopped()

    def _get_content(msg) -> str:
        # Anthropic models return content as a list of blocks; OpenAI returns a plain string.
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        return content or ""

    # The final AIMessage can be empty for reasoning models (reasoning tokens are internal).
    # Walk backwards to find the last message that actually has visible content.
    output = next(
        (
            _get_content(msg)
            for msg in reversed(final_state["messages"])
            if type(msg).__name__ == "AIMessage" and _get_content(msg).strip()
        ),
        "",
    )

    if not output.strip():
        raise ValueError("Agent returned empty output — it did not write any files or produce a summary")

    logger.info(f"Task {task['id']} agent returned {len(output)} chars")

    # ── LLM-as-judge ─────────────────────────────────────────────────
    verdict = await _judge_task(task, output)
    logger.info(
        f"Judge verdict for {task['id']}: score={verdict.score}, passed={verdict.passed}, reason={verdict.reason}"
    )
    if not verdict.passed:
        raise ValueError(f"Judge rejected output (score={verdict.score}/10): {verdict.reason}")

    return output
