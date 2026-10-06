"""Retry policy for LLM and embedding calls: transient errors only, exponential backoff + jitter."""

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from langchain_core.embeddings import Embeddings

from kartrix.config import RetrySettings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# 408 timeout, 409 conflict, 425 too early, 429 rate limited, 5xx server-side.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# langchain-nvidia-ai-endpoints raises bare Exception("[429] Too Many Requests\n...").
_STATUS_PREFIX = re.compile(r"^\s*\[(\d{3})\]")

_TRANSIENT_NAMES = frozenset({
    "TimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout", "ServerTimeoutError",
    "ConnectionError", "ClientConnectionError", "ServerDisconnectedError",
    "APITimeoutError", "APIConnectionError", "RemoteProtocolError",
})


def status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from SDK exceptions (openai, httpx, requests, NVIDIA)."""
    for candidate in (getattr(exc, "status_code", None), getattr(getattr(exc, "response", None), "status_code", None)):
        if isinstance(candidate, int):
            return candidate
    match = _STATUS_PREFIX.match(str(exc))
    return int(match.group(1)) if match else None


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying; auth/validation/not-found errors are not."""
    status = status_code_of(exc)
    if status is not None:
        return status in TRANSIENT_STATUS
    return any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__) or isinstance(exc, asyncio.TimeoutError)


def is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    return any("Timeout" in cls.__name__ for cls in type(exc).__mro__) or status_code_of(exc) == 408


def should_retry_model_call(exc: Exception) -> bool:
    """``retry_on`` hook for ModelRetryMiddleware: classify and log every model failure.

    Timeouts are *not* retried here: a model that just hung for ``llm.timeout``
    seconds is usually overloaded, so moving on to the next fallback is faster.
    Rate limits, 5xx and dropped connections are retried with backoff.
    """
    transient = is_transient(exc) and not is_timeout(exc)
    logger.warning(
        "Model call failed",
        extra={"error_type": type(exc).__name__, "status": status_code_of(exc), "will_retry": transient, "error": str(exc)[:200]},
    )
    return transient


def backoff_delay(attempt: int, policy: RetrySettings) -> float:
    """Delay before retry number ``attempt`` (0-based): exponential, capped, full jitter."""
    ceiling = min(policy.max_delay, policy.initial_delay * policy.backoff_factor**attempt)
    return random.uniform(ceiling / 2, ceiling)


def call_with_retry(fn: Callable[[], T], policy: RetrySettings, what: str) -> T:
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= policy.max_retries or not is_transient(exc):
                raise
            delay = backoff_delay(attempt, policy)
            logger.warning("Transient error, retrying", extra={"call": what, "attempt": attempt + 1, "delay_s": round(delay, 2), "error": str(exc)[:200]})
            time.sleep(delay)
    raise AssertionError("unreachable")


async def acall_with_retry(fn: Callable[[], Awaitable[T]], policy: RetrySettings, what: str) -> T:
    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt >= policy.max_retries or not is_transient(exc):
                raise
            delay = backoff_delay(attempt, policy)
            logger.warning("Transient error, retrying", extra={"call": what, "attempt": attempt + 1, "delay_s": round(delay, 2), "error": str(exc)[:200]})
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class RetryingEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings with the transient-error retry policy."""

    def __init__(self, inner: Embeddings, policy: RetrySettings) -> None:
        self.inner = inner
        self.policy = policy

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return call_with_retry(lambda: self.inner.embed_documents(texts), self.policy, "embed_documents")

    def embed_query(self, text: str) -> list[float]:
        return call_with_retry(lambda: self.inner.embed_query(text), self.policy, "embed_query")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await acall_with_retry(lambda: self.inner.aembed_documents(texts), self.policy, "aembed_documents")

    async def aembed_query(self, text: str) -> list[float]:
        return await acall_with_retry(lambda: self.inner.aembed_query(text), self.policy, "aembed_query")
