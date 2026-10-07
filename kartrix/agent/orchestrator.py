from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kartrix.agent.reliability import final_text
from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.security.approvals import Approver, run_agent
from kartrix.security.budget import current as current_budget

logger = get_logger(__name__)


@dataclass(frozen=True)
class QueryAnswer:
    text: str
    cached: bool = False  # served from the semantic cache: the agent didn't run


async def handle_query(
    agent: Any,
    question: str,
    thread_id: str,
    approver: Approver,
    semantic_cache: Any = None,
    cache_domain: str | None = None,
) -> QueryAnswer:
    """Answer one chat request with the agent; agent errors propagate to the caller.

    When semantic_cache/cache_domain are provided, a cache hit returns the
    stored answer directly and skips the agent (and every tool call it would
    have made, including search_codebase) entirely. A miss falls through to
    the normal agent call and stores the fresh answer for next time.
    Commands that need approval pause the agent until ``approver`` answers.
    """
    logger.info(f"Handling query for session {thread_id}: {question}")
    model = settings.llm.model

    if semantic_cache is not None and cache_domain is not None:
        try:
            cached_response = await semantic_cache.get(question, domain=cache_domain, model=model)
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed, falling back to agent: {e}")
            cached_response = None
        if cached_response is not None:
            logger.info("Semantic cache HIT - skipping agent/tool calls")
            return QueryAnswer(cached_response, cached=True)

    agent_config = {"configurable": {"thread_id": thread_id}}
    response = await run_agent(agent, {"messages": [{"role": "user", "content": question}]}, agent_config, approver)
    answer = final_text(response.get("messages", []))

    run = current_budget()
    stopped = run is not None and run.stop_reason is not None  # never cache a "Stopped: …" answer
    # Only answers to read-only questions are cached: a cached reply to "fix X" would claim the fix
    # without making it. The single-loop agent has no route, so its answers are never cached.
    question_only = response.get("route") == "question"
    if semantic_cache is not None and cache_domain is not None and answer and not stopped and question_only:
        try:
            ttl = settings.semantic_cache.ttl
            await semantic_cache.put(question, answer, domain=cache_domain, model=model, ttl=ttl)
        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")

    return QueryAnswer(answer)
