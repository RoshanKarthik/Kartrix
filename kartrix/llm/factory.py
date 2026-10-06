"""LLM and embedding providers.

Default: NVIDIA NIM. Fallbacks (config ``llm.fallbacks``, in order): another NIM
model, then Hugging Face Inference Providers via its OpenAI-compatible router.
Agents get resilience through middleware (``get_model_middleware``): each model
call is retried on transient errors with exponential backoff, and if the primary
still fails the fallback models are tried in turn.
API keys come only from env vars (NVIDIA_API_KEY, HF_TOKEN, ...).
"""

import os
import warnings
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from kartrix.config import settings
from kartrix.llm.retry import RetryingEmbeddings, should_retry_model_call
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

# The NVIDIA client warns for catalogue models it has no type metadata for (e.g.
# nemotron-3-embed-1b); they work fine, so keep the REPL free of that noise.
warnings.filterwarnings("ignore", message=r"Found .* in available_models, but type is unknown", category=UserWarning)

HF_ROUTER_URL = "https://router.huggingface.co/v1"

Role = Literal["main", "judge"]


class ProviderConfigError(RuntimeError):
    """Raised when a provider's API key is missing."""


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderConfigError(f"{name} is not set — add it to .env (see .env.example)")
    return value


def _build_chat_model(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    timeout = settings.llm.timeout
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
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, timeout=timeout, max_retries=0, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, timeout=timeout, max_retries=0, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_chat_model(role: Role = "main", **kwargs: Any) -> BaseChatModel:
    """The primary model for a role. ``kwargs`` (temperature, max_tokens, ...) are passed through."""
    cfg = settings.llm
    model = cfg.effective_judge_model if role == "judge" else cfg.model
    logger.info("Using LLM", extra={"role": role, "provider": cfg.provider, "model": model})
    return _build_chat_model(cfg.provider, model, **kwargs)


def get_fallback_chat_models(**kwargs: Any) -> list[BaseChatModel]:
    """Fallback models in configured order (may be empty)."""
    return [_build_chat_model(fb.provider, fb.model, **kwargs) for fb in settings.llm.fallbacks]


def get_model_middleware(**kwargs: Any) -> list[AgentMiddleware]:
    """Middleware for ``create_agent``: fallback (outer) around retry (inner).

    Order matters — the retry wraps each model attempt, so the primary is retried
    with backoff first, then each fallback model in turn gets the same retry budget.
    """
    policy = settings.llm.retry
    middleware: list[AgentMiddleware] = []
    fallbacks = get_fallback_chat_models(**kwargs)
    if fallbacks:
        middleware.append(ModelFallbackMiddleware(*fallbacks))
    middleware.append(
        ModelRetryMiddleware(
            max_retries=policy.max_retries,
            retry_on=should_retry_model_call,
            on_failure="error",  # surface the error so the fallback middleware can act on it
            initial_delay=policy.initial_delay,
            backoff_factor=policy.backoff_factor,
            max_delay=policy.max_delay,
            jitter=True,
        )
    )
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
