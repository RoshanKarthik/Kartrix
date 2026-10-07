"""Model fallback with a circuit breaker.

The chain is the primary model, then ``llm.fallbacks`` in order; each model is retried on transient
errors by ``ModelRetryMiddleware`` inside it. On top of LangChain's plain fallback this adds:

- **Circuit breaker.** A model whose provider answers 401/402/403/404 (bad key, no credits, no
  access, model gone) won't answer the next call either: it is skipped for the rest of the process
  instead of costing a request (and a timeout) on every call. A model that timed out is skipped for
  ``COOL_DOWN`` seconds by later calls.
- **Second chance for the primary.** Free-tier timeouts are usually short outages: when every model
  failed and the primary only timed out, it is tried once more.
- **One clear error.** When everything fails, :class:`ModelUnavailableError` says what happened to
  each model, instead of re-raising whatever the last fallback said.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langgraph.errors import GraphBubbleUp

from kartrix.llm.retry import is_timeout, status_code_of
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

PERMANENT_STATUS = frozenset({401, 402, 403, 404})
COOL_DOWN = 120.0  # seconds a timed-out model is skipped by later calls
RETRY_PAUSE = 2.0  # before the primary's second chance

_STATUS_TEXT = {401: "invalid API key", 402: "out of credits", 403: "no access", 404: "model not found"}


class ModelUnavailableError(RuntimeError):
    """Every model in the chain failed (or is switched off by the circuit breaker)."""


@dataclass
class _Open:
    until: float  # monotonic time; inf = for the rest of the process
    reason: str


_circuit: dict[str, _Open] = {}
_lock = threading.Lock()


def model_name(model: Any) -> str:
    return str(getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__)


def _describe(exc: BaseException) -> str:
    status = status_code_of(exc)
    if status in _STATUS_TEXT:
        return f"{_STATUS_TEXT[status]} ({status})"
    if is_timeout(exc):
        return "timed out"
    text = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _skip_reason(name: str) -> str | None:
    with _lock:
        state = _circuit.get(name)
        if state is None:
            return None
        if state.until <= time.monotonic():
            del _circuit[name]
            return None
        return state.reason


def _record_failure(name: str, exc: BaseException) -> None:
    status = status_code_of(exc)
    with _lock:
        if status in PERMANENT_STATUS:
            _circuit[name] = _Open(float("inf"), f"{_describe(exc)} — skipped until Kartrix restarts")
        elif is_timeout(exc):
            _circuit[name] = _Open(time.monotonic() + COOL_DOWN, "timed out recently")


def _record_success(name: str) -> None:
    with _lock:
        _circuit.pop(name, None)


def reset_circuit() -> None:
    """Forget every failure (tests, or after the user changed keys)."""
    with _lock:
        _circuit.clear()


def _request_for(request: ModelRequest[Any], model: BaseChatModel, primary: bool) -> ModelRequest[Any]:
    if primary:
        return request
    try:  # drop provider-specific settings (e.g. prompt-cache markers) the fallback can't accept
        from langchain.agents.middleware.model_fallback import _sanitize_request_for_fallback

        request = _sanitize_request_for_fallback(request, model)
    except ImportError:
        pass
    return request.override(model=model)


class _Attempts:
    def __init__(self) -> None:
        self.notes: list[str] = []
        self.primary_timed_out = False

    def error(self) -> ModelUnavailableError:
        return ModelUnavailableError("no model answered — " + "; ".join(self.notes))


def _chain(
    request: ModelRequest[Any], fallbacks: Callable[[], list[BaseChatModel]]
) -> Iterator[tuple[BaseChatModel, bool]]:
    """The primary, then the fallbacks — built only when the primary failed (their SDK imports are slow)."""
    yield request.model, True
    for model in fallbacks():
        yield model, False


def _failed(attempts: _Attempts, name: str, primary: bool, exc: Exception) -> None:
    _record_failure(name, exc)
    attempts.notes.append(f"{name}: {_describe(exc)}")
    if primary and is_timeout(exc):
        attempts.primary_timed_out = True
    logger.warning("Model failed; trying the next one", extra={"model": name, "error": _describe(exc)})


def call_chain(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Any],
    fallbacks: Callable[[], list[BaseChatModel]],
) -> Any:
    attempts = _Attempts()
    for model, primary in _chain(request, fallbacks):
        name = model_name(model)
        if (why := _skip_reason(name)) is not None:
            attempts.notes.append(f"{name}: {why}")
            continue
        try:
            response = handler(_request_for(request, model, primary))
        except GraphBubbleUp:
            raise
        except Exception as e:
            _failed(attempts, name, primary, e)
            continue
        _record_success(name)
        return response
    if attempts.primary_timed_out:
        time.sleep(RETRY_PAUSE)
        try:
            response = handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            attempts.notes.append(f"{model_name(request.model)} (second try): {_describe(e)}")
        else:
            _record_success(model_name(request.model))
            return response
    raise attempts.error()


async def acall_chain(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[Any]],
    fallbacks: Callable[[], list[BaseChatModel]],
) -> Any:
    attempts = _Attempts()
    for model, primary in _chain(request, fallbacks):
        name = model_name(model)
        if (why := _skip_reason(name)) is not None:
            attempts.notes.append(f"{name}: {why}")
            continue
        try:
            response = await handler(_request_for(request, model, primary))
        except GraphBubbleUp:
            raise
        except Exception as e:  # CancelledError is a BaseException and passes through
            _failed(attempts, name, primary, e)
            continue
        _record_success(name)
        return response
    if attempts.primary_timed_out:
        await asyncio.sleep(RETRY_PAUSE)
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            attempts.notes.append(f"{model_name(request.model)} (second try): {_describe(e)}")
        else:
            _record_success(model_name(request.model))
            return response
    raise attempts.error()
