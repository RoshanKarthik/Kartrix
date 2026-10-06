from langchain.agents import create_agent

from kartrix.agent.tools import search_codebase
from kartrix.llm.factory import get_llm, get_model_middleware
from kartrix.mcp.mcp_client import get_mcp_tools
from kartrix.observability.logger import get_logger
from kartrix.skills.skill_tools import build_skills_prompt, load_skill
from kartrix.tools.terminal_tools import run_command, run_in_directory

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""


async def build_agent(checkpointer):
    """Create and return a LangChain agent with persistent memory."""
    llm = get_llm()
    mcp_tools = await get_mcp_tools()

    skills_prompt = build_skills_prompt()
    full_prompt = SYSTEM_PROMPT
    if skills_prompt:
        full_prompt = SYSTEM_PROMPT + "\n\n" + skills_prompt

    tools = [
        search_codebase,
        load_skill,
        run_command,
        run_in_directory,
        *mcp_tools,
    ]

    return create_agent(
        llm,
        tools=tools,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
        middleware=get_model_middleware(),
    )
