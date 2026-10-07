"""Retrying model calls — with patience for rate limits.

LangChain's ``ModelRetryMiddleware`` uses one backoff for every error. A free-tier provider's rate limit
(429) is per minute, so a few seconds of backoff just collects more 429s and the run fails over to the
fallback models (or dies, when they are down too). Here:

- **429** is retried up to ``llm.retry.rate_limit_retries`` times, waiting what the provider asks for
  (``Retry-After``) or else an exponential delay up to ``rate_limit_max_delay`` (60 s by default);
- other transient errors (5xx, dropped connections) get the usual ``max_retries`` with backoff + jitter;
- timeouts are not retried here — the fallback chain moves on and gives the primary a second chance later;
- waiting is logged, and a stopped run (budget, kill switch) is never kept waiting.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langgraph.errors import GraphBubbleUp

from kartrix.config import settings
from kartrix.llm.retry import backoff_delay, should_retry_model_call, status_code_of
from kartrix.observability.logger import get_logger
from kartrix.security.budget import current as current_budget

logger = get_logger(__name__)


def retry_after(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait (``Retry-After`` header), if it said."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def delay_for(exc: Exception, attempt: int) -> float | None:
    """How long to wait before retry ``attempt`` (0-based) after ``exc``; None = don't retry."""
    policy = settings.llm.retry
    if status_code_of(exc) == 429:
        if attempt >= policy.rate_limit_retries:
            return None
        asked = retry_after(exc)
        if asked is not None:
            return min(asked + random.uniform(0, 1), policy.rate_limit_max_delay)  # noqa: S311 — jitter
        ceiling = min(policy.rate_limit_max_delay, 5.0 * 2**attempt)
        return random.uniform(ceiling / 2, ceiling)  # noqa: S311 — jitter
    if attempt >= policy.max_retries or not should_retry_model_call(exc):
        return None
    return backoff_delay(attempt, policy)


def _stopped() -> bool:
    budget = current_budget()
    return budget is not None and budget.exhausted() is not None


class RateAwareRetryMiddleware(AgentMiddleware):
    """Retries one model's calls (the fallback chain wraps it, so each model gets its own retries)."""

    def wrap_model_call(self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Any]) -> Any:
        attempt = 0
        while True:
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                delay = delay_for(exc, attempt)
                if delay is None or _stopped():
                    raise
                _log(exc, attempt, delay)
                time.sleep(delay)
                attempt += 1

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[Any]]
    ) -> Any:
        attempt = 0
        while True:
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                delay = delay_for(exc, attempt)
                if delay is None or _stopped():
                    raise
                _log(exc, attempt, delay)
                await asyncio.sleep(delay)
                attempt += 1


def _log(exc: Exception, attempt: int, delay: float) -> None:
    logger.warning(
        "Model call failed; retrying",
        extra={
            "status": status_code_of(exc),
            "attempt": attempt + 1,
            "delay_s": round(delay, 1),
            "error": str(exc)[:200],
        },
    )
