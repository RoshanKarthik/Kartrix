"""The second retrieval stage: reranked order, and the fused order kept when the reranker fails."""

from __future__ import annotations

import time

import pytest

from kartrix.config import settings
from kartrix.context import rerank as rr


def _chunks(*names: str) -> list[dict]:
    return [{"content": f"code of {n}", "source": f"src/{n}.py", "name": n} for n in names]


def _by_keyword(query: str, passages: list[str]) -> list[float]:
    return [float(p.count(query)) for p in passages]


async def test_reorders_by_score_and_cuts_to_k():
    rr.set_reranker(lambda: _by_keyword)
    chunks = _chunks("alpha", "beta", "beta_beta", "gamma")
    out = await rr.rerank("beta", chunks, k=2)
    assert [c["name"] for c in out] == ["beta_beta", "beta"]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


async def test_ties_keep_the_first_stage_order():
    rr.set_reranker(lambda: lambda q, ps: [0.0] * len(ps))
    out = await rr.rerank("x", _chunks("a", "b", "c"), k=3)
    assert [c["name"] for c in out] == ["a", "b", "c"]


async def test_reads_path_and_symbol_and_skips_empty_passages():
    seen: list[list[str]] = []

    def capture(query: str, passages: list[str]) -> list[float]:
        seen.append(passages)
        return [1.0] * len(passages)

    rr.set_reranker(lambda: capture)
    chunks = [*_chunks("a"), {"content": "  ", "source": "src/empty.py", "name": None}, *_chunks("b")]
    out = await rr.rerank("q", chunks, k=3)
    assert seen == [["src/a.py · a\ncode of a", "src/b.py · b\ncode of b"]]
    assert out[-1]["source"] == "src/empty.py"


async def test_failure_keeps_fused_order_and_pauses(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def broken(query: str, passages: list[str]) -> list[float]:
        nonlocal calls
        calls += 1
        raise RuntimeError("[429] Too Many Requests")

    rr.set_reranker(lambda: broken)
    chunks = _chunks("a", "b", "c")
    assert [c["name"] for c in await rr.rerank("q", chunks, k=2)] == ["a", "b"]
    assert [c["name"] for c in await rr.rerank("q", chunks, k=2)] == ["a", "b"]
    assert calls == 1  # paused after the first failure
    later = time.monotonic() + settings.retrieval.rerank.cooldown_s + 1
    monkeypatch.setattr(rr.time, "monotonic", lambda: later)
    await rr.rerank("q", chunks, k=2)
    assert calls == 2  # tried again after the cooldown


async def test_timeout_keeps_fused_order(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.retrieval.rerank, "timeout", 0.05)

    def slow(query: str, passages: list[str]) -> list[float]:
        time.sleep(0.5)
        return [1.0] * len(passages)

    rr.set_reranker(lambda: slow)
    assert [c["name"] for c in await rr.rerank("q", _chunks("a", "b"), k=1)] == ["a"]


async def test_a_missing_key_is_a_failure_not_a_crash(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    rr.set_reranker(None)  # the real factory: fails on the missing key before any request
    assert [c["name"] for c in await rr.rerank("q", _chunks("a", "b"), k=1)] == ["a"]


def test_fusion_with_the_first_stage_rank():
    """stage1_weight 0: the reranker's order; higher weights let a strong first-stage hit keep its place."""
    chunks = [{"name": n, "rerank_score": s} for n, s in (("a", 0.1), ("b", 0.9), ("c", 0.5))]
    assert [c["name"] for c in rr.fuse(chunks, 0.0)] == ["b", "c", "a"]
    assert [c["name"] for c in rr.fuse(chunks, 1.0)] == ["b", "a", "c"]  # a: 1st in stage 1, 3rd in stage 2
