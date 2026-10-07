"""Second-stage reranking of retrieved chunks with a cross-encoder (``retrieval.rerank``).

The first stage (``retrievers.pg_hybrid``) is cheap and recall-oriented: it fuses pgvector and full-text
ranks over ``rerank.candidates`` chunks. The reranker (NIM ``llama-nemotron-rerank``) reads the query and
each passage *together*, which embeddings can't, and re-orders them by relevance.

Best effort: a failure (timeout, 429, outage, a retired model) keeps the first-stage order and pauses
reranking for ``cooldown_s``, so a dead endpoint costs one timeout, not one per search.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Protocol

from kartrix.config import settings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    def __call__(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage (higher = more relevant), in the passages' order."""
        ...


def _nim_reranker() -> Reranker:
    from langchain_core.documents import Document
    from langchain_nvidia_ai_endpoints import NVIDIARerank

    from kartrix.llm.factory import _require_env

    cfg = settings.retrieval.rerank
    client = NVIDIARerank(model=cfg.model, api_key=_require_env("NVIDIA_API_KEY"), truncate="END", top_n=10_000)

    def score(query: str, passages: list[str]) -> list[float]:
        docs = [Document(page_content=p, metadata={"i": i}) for i, p in enumerate(passages)]
        scores = [0.0] * len(passages)
        for d in client.compress_documents(docs, query):
            scores[d.metadata["i"]] = float(d.metadata["relevance_score"])
        return scores

    return score


_factory: Callable[[], Reranker] = _nim_reranker
_reranker: Reranker | None = None
_paused_until = 0.0


def set_reranker(factory: Callable[[], Reranker] | None) -> None:
    """Replace the reranker (tests); ``None`` restores the NIM one. Also clears a cooldown."""
    global _factory, _reranker, _paused_until
    _factory = factory or _nim_reranker
    _reranker, _paused_until = None, 0.0


def passage(chunk: dict[str, Any], max_chars: int) -> str:
    """What the reranker reads: the file and symbol, then the code (cut to ``max_chars``)."""
    head = chunk["source"] + (f" · {chunk['name']}" if chunk.get("name") else "")
    return f"{head}\n{chunk['content']}"[:max_chars]


def fuse(chunks: list[dict[str, Any]], stage1_weight: float, rrf_k: int = 60) -> list[dict[str, Any]]:
    """Order ``chunks`` (given in first-stage order, with ``rerank_score``) by reciprocal rank fusion of the
    reranker's rank and, weighted by ``stage1_weight``, the first-stage rank (0 = the reranker alone)."""
    by_score = sorted(range(len(chunks)), key=lambda i: (-chunks[i]["rerank_score"], i))
    rank2 = {i: r for r, i in enumerate(by_score, 1)}
    fused = {i: 1 / (rrf_k + rank2[i]) + stage1_weight / (rrf_k + i + 1) for i in range(len(chunks))}
    return [chunks[i] for i in sorted(fused, key=lambda i: (-fused[i], i))]


async def rerank(query: str, chunks: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """The top ``k`` of ``chunks`` by reranker score (``rerank_score`` added); the first ``k`` on failure."""
    global _reranker, _paused_until
    cfg = settings.retrieval.rerank
    if len(chunks) <= 1 or time.monotonic() < _paused_until:
        return chunks[:k]
    # The endpoint rejects empty passages; they can't be relevant anyway, so they go last.
    usable = [c for c in chunks if c["content"].strip()]
    empty = [c for c in chunks if not c["content"].strip()]
    started = time.perf_counter()
    try:
        if _reranker is None:
            _reranker = _factory()
        reranker = _reranker
        passages = [passage(c, cfg.max_chars) for c in usable]
        scores = await asyncio.wait_for(asyncio.to_thread(reranker, query, passages), cfg.timeout)
    except Exception as e:  # any failure: keep the first-stage order
        _paused_until = time.monotonic() + cfg.cooldown_s
        logger.warning(
            "Reranking failed — keeping the fused order",
            extra={"error": f"{type(e).__name__}: {e}"[:300], "paused_s": cfg.cooldown_s},
        )
        return chunks[:k]
    out = [{**c, "rerank_score": round(s, 4)} for s, c in zip(scores, usable, strict=True)]
    out = fuse(out, cfg.stage1_weight) + empty
    logger.info(
        "Reranked chunks",
        extra={"candidates": len(chunks), "k": k, "ms": round((time.perf_counter() - started) * 1000)},
    )
    return out[:k]
