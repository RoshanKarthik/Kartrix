from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from kartrix.agent.tools import search_codebase
from kartrix.llm.factory import get_llm, get_model_middleware
from kartrix.observability.logger import get_logger
from kartrix.security.approvals import ApprovalMiddleware
from kartrix.security.audit import AuditMiddleware
from kartrix.security.budget import BudgetMiddleware
from kartrix.security.injection import SECURITY_RULES, ContentGuardMiddleware
from kartrix.skills.skill_tools import build_skills_prompt, load_skill
from kartrix.tools.filesystem_tools import READ_TOOLS, WRITE_TOOLS
from kartrix.tools.terminal_tools import run_command

logger = get_logger(__name__)

SYSTEM_PROMPT = f"""You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Use grep, glob and read_file to look at exact code, and edit_file for small changes to existing files.
File paths are relative to the repository root; files outside it, secrets and .git are off limits.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly.

{SECURITY_RULES}"""

# Kartrix's own tools; MCP servers may not register tools with these names.
NATIVE_TOOLS = [search_codebase, load_skill, *READ_TOOLS, *WRITE_TOOLS, run_command]


def agent_middleware(approval: ApprovalMiddleware | None, **model_kwargs: Any) -> list[AgentMiddleware]:
    """The middleware stack every tool-using agent gets (first = outermost).

    Approval runs outside the audit wrapper, so the audit row shows the command that actually ran.
    The budget check runs inside it, so refused calls are audited, and inside the retry/fallback
    wrappers, so every model attempt is counted. The content guard runs innermost, so the audit row
    records its findings."""
    return [
        *get_model_middleware(**model_kwargs),
        *([approval] if approval is not None else []),
        AuditMiddleware(),
        BudgetMiddleware(),
        ContentGuardMiddleware(),
    ]


def build_agent(checkpointer, mcp_tools: list | None = None):
    """Create the chat agent. It is rebuilt only when MCP servers connect/disconnect or skills
    are trusted, so the system prompt stays identical between those events (prompt caching)."""
    llm = get_llm()

    full_prompt = SYSTEM_PROMPT
    skills_prompt = build_skills_prompt()
    if skills_prompt:
        full_prompt += "\n\n" + skills_prompt

    return create_agent(
        llm,
        tools=[*NATIVE_TOOLS, *(mcp_tools or [])],
        system_prompt=full_prompt,
        checkpointer=checkpointer,
        middleware=agent_middleware(ApprovalMiddleware()),
    )
