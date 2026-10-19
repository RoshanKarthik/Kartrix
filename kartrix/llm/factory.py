"""LLM and embedding providers.

Default: NVIDIA NIM. Fallbacks (config ``llm.fallbacks``, in order): another NIM
model, then Hugging Face Inference Providers via its OpenAI-compatible router.
Agents get resilience through middleware (``get_model_middleware``): each model
call is retried on transient errors with exponential backoff, and if the primary
still fails the fallback models are tried in turn.
API keys come only from env vars (NVIDIA_API_KEY, HF_TOKEN, ...).
"""

import functools
import os
import threading
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from kartrix.config import LLMModelRef, settings
from kartrix.llm.catalog import PROVIDER_KEYS as _PROVIDER_KEYS
from kartrix.llm.fallback import acall_chain, call_chain
from kartrix.llm.model_retry import RateAwareRetryMiddleware
from kartrix.llm.retry import RetryingEmbeddings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

# The NVIDIA client warns for catalogue models it has no type metadata for (e.g.
# nemotron-3-embed-1b); they work fine, so keep the REPL free of that noise.
warnings.filterwarnings("ignore", message=r"Found .* in available_models, but type is unknown", category=UserWarning)

HF_ROUTER_URL = "https://router.huggingface.co/v1"
# Providers reached through their OpenAI-compatible endpoints: base URL and the env var with the key.
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
}

Role = Literal["main", "judge", "router"]


class ProviderConfigError(RuntimeError):
    """Raised when a provider's API key is missing."""


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderConfigError(f"{name} is not set — run `kartrix keys set {name}` (or add it to .env)")
    return value


# The model the user picked for the main role (``select_main_model``); None = ``llm.provider`` / ``llm.model``.
# Router and judge stay the configured cheap models whatever is picked.
_selected_main: LLMModelRef | None = None


def select_main_model(ref: LLMModelRef | None) -> None:
    """Use ``ref`` for the main role from now on in this process (None: back to the configured default).
    Agents already built keep their model: rebuild them (``CoreSession.set_model`` does)."""
    global _selected_main
    _selected_main = ref
    logger.info("Main model selected", extra={"model": ref.model if ref else settings.llm.model})


def main_model() -> LLMModelRef:
    return _selected_main or LLMModelRef(provider=settings.llm.provider, model=settings.llm.model)


# Current Claude models reject sampling parameters (400); callers ask for temperature=0 for determinism,
# which those models don't offer anyway.
_ANTHROPIC_UNSUPPORTED = ("temperature", "top_p", "top_k")


def _build_chat_model(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    timeout = settings.llm.timeout
    kwargs.setdefault("max_tokens", settings.llm.max_output_tokens)
    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        return ChatNVIDIA(model=model, api_key=_require_env("NVIDIA_API_KEY"), timeout=timeout, **kwargs)
    if provider == "huggingface":
        from langchain_openai import ChatOpenAI

        # Retries are handled by our middleware, so the SDK's own retries are off.
        return ChatOpenAI(
            model=model,
            base_url=HF_ROUTER_URL,
            api_key=_require_env("HF_TOKEN"),
            timeout=timeout,
            max_retries=0,
            **kwargs,
        )
    if provider in OPENAI_COMPATIBLE:
        from langchain_openai import ChatOpenAI

        base_url, key = OPENAI_COMPATIBLE[provider]
        headers = {"X-Title": "Kartrix"} if provider == "openrouter" else None  # app name in OpenRouter's logs
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=_require_env(key),
            timeout=timeout,
            max_retries=0,
            default_headers=headers,
            **kwargs,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, timeout=timeout, max_retries=0, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        for name in _ANTHROPIC_UNSUPPORTED:
            kwargs.pop(name, None)
        return ChatAnthropic(model=model, timeout=timeout, max_retries=0, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_chat_model(role: Role = "main", **kwargs: Any) -> BaseChatModel:
    """The primary model for a role. ``kwargs`` (temperature, max_tokens, ...) are passed through."""
    cfg = settings.llm
    if role == "main":
        provider, model = main_model().provider, main_model().model
    else:
        provider = cfg.provider
        model = cfg.effective_judge_model if role == "judge" else cfg.effective_router_model
    logger.info("Using LLM", extra={"role": role, "provider": provider, "model": model})
    return _build_chat_model(provider, model, **kwargs)


def main_response_format(schema: Any) -> Any:
    """``response_format`` for an agent on the main model. Claude models get native structured output:
    LangChain falls back to a forced tool call for models it doesn't know (e.g. Claude Haiku 5.5), and current
    Claude models reject forced tool use (or combine it badly with their always-on thinking)."""
    if main_model().provider == "anthropic":
        from langchain.agents.structured_output import ProviderStrategy

        return ProviderStrategy(schema)
    return schema


def get_fallback_chat_models(**kwargs: Any) -> list[BaseChatModel]:
    """Fallback models in configured order (may be empty)."""
    return [_build_chat_model(fb.provider, fb.model, **kwargs) for fb in settings.llm.fallbacks]


class LazyModelFallbackMiddleware(ModelFallbackMiddleware):
    """Fallback models built on first use, tried through the circuit-breaking chain in
    :mod:`kartrix.llm.fallback` (skips dead providers, second chance for a timed-out primary, one
    clear error when every model fails).

    Building a client means importing its SDK (``langchain_openai`` alone takes seconds on a
    cold start) for a model that is only needed when the primary fails — so it waits until then."""

    def __init__(self, factories: list[Callable[[], BaseChatModel]]) -> None:
        AgentMiddleware.__init__(self)
        self._factories = factories
        self._models: list[BaseChatModel] | None = None
        self._build_lock = threading.Lock()

    @property
    def models(self) -> list[BaseChatModel]:
        with self._build_lock:
            if self._models is None:
                self._models = [factory() for factory in self._factories]
            return self._models

    @models.setter
    def models(self, value: list[BaseChatModel]) -> None:
        self._models = value

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return call_chain(request, handler, lambda: self.models)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await acall_chain(request, handler, lambda: self.models)


def available_fallbacks() -> list[LLMModelRef]:
    """The configured fallback models whose provider key is set. The others are left out of the chain (logged once
    per call) — a fresh install with only the required key must start, and a model without its key could never
    answer anyway."""
    usable = []
    for fb in settings.llm.fallbacks:
        key = _PROVIDER_KEYS.get(fb.provider)
        if key is None or os.environ.get(key, "").strip():
            usable.append(fb)
        else:
            logger.info("Fallback model skipped: no key", extra={"model": fb.model, "key": key})
    return usable


def get_model_middleware(**kwargs: Any) -> list[AgentMiddleware]:
    """Middleware for ``create_agent``: fallback (outer) around retry (inner).

    Order matters — the retry wraps each model attempt, so the primary is retried
    with backoff first, then each fallback model in turn gets the same retry budget.
    """
    middleware: list[AgentMiddleware] = []
    fallbacks = available_fallbacks()
    # Always added: the circuit breaker and the clear error also matter without fallbacks.
    middleware.append(
        LazyModelFallbackMiddleware(
            [functools.partial(_build_chat_model, fb.provider, fb.model, **kwargs) for fb in fallbacks]
        )
    )
    # Per model: 429 waits as long as the provider asks (or up to a minute), other transient errors back off
    # briefly; the error then reaches the fallback chain above.
    middleware.append(RateAwareRetryMiddleware())
    return middleware


def get_llm() -> BaseChatModel:
    """Backwards-compatible alias for the main chat model."""
    return get_chat_model("main")


def get_embedder() -> Embeddings:
    """Embedding model wrapped with retries (no fallback: vector spaces must not mix)."""
    cfg = settings.embeddings
    logger.info("Using embeddings", extra={"provider": cfg.provider, "model": cfg.model, "dims": cfg.dims})
    inner: Embeddings
    if cfg.provider == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        # truncate=END: over-long chunks are cut instead of failing the whole batch.
        inner = NVIDIAEmbeddings(
            model=cfg.model, api_key=_require_env("NVIDIA_API_KEY"), truncate="END", timeout=cfg.timeout
        )
    elif cfg.provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        inner = HuggingFaceEmbeddings(model_name=cfg.model)
    else:
        from langchain_openai import OpenAIEmbeddings

        inner = OpenAIEmbeddings(model=cfg.model, max_retries=0)
    return RetryingEmbeddings(inner, cfg.retry)
