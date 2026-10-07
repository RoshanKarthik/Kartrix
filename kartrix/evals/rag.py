"""``kartrix eval rag``: retrieval quality per mode, then answer quality judged by an LLM.

1. Every fixture repo the questions use is materialised and indexed with the configured embedder
   (incremental: later runs only pay for changes, i.e. a new pin or chunker version).
2. Each question runs through every retrieval mode; the ranked chunks are scored against the
   labelled files and symbols (:mod:`kartrix.evals.metrics`), with latency and the context tokens the
   top ``retrieval.top_k`` chunks would cost.
3. For the answer modes (default: the configured ``retrieval.mode``) the main model answers from those
   chunks, and DeepEval's faithfulness, answer relevancy, contextual precision and contextual recall
   judge the answer and the context (reference answer = the golden answer).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kartrix.config import settings
from kartrix.evals.datasets import GoldenQuestion, load_golden, load_repos
from kartrix.evals.fixtures import ensure_repo
from kartrix.evals.metrics import K_VALUES, distinct_files, mean, percentile, retrieval_scores
from kartrix.evals.retrievers import INDEX_MODES, Chunk, get_retriever
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

DEPTH = 20  # chunks retrieved per query (enough for ~10 distinct files)

ANSWER_PROMPT = """\
You answer questions about a code repository using only the code excerpts below.
Name the files (and functions/classes) your answer relies on. If the excerpts do not contain the
answer, say so instead of guessing. Keep the answer short: a few sentences.

Question: {question}

Code excerpts:
{context}
"""


@dataclass
class RagOptions:
    modes: tuple[str, ...]
    answer_modes: tuple[str, ...] = ()
    judge: bool = True
    quick: bool = False
    ids: list[str] | None = None
    concurrency: int = 4
    refresh_fixtures: bool = False
    progress: Any = None  # callable(str) for progress lines


@dataclass
class _Run:
    options: RagOptions
    roots: dict[str, Path] = field(default_factory=dict)
    retrieved: dict[tuple[str, str], list[Chunk]] = field(default_factory=dict)


def format_chunk(chunk: Chunk) -> str:
    name = f" — {chunk['name']}" if chunk.get("name") else ""
    return f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']}){name}\n{chunk['content']}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class CachingEmbeddings:
    """Embeds each distinct query once per eval run (dense and hybrid would embed it twice)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self._pending: dict[str, asyncio.Future[list[float]]] = {}

    async def aembed_query(self, text: str) -> list[float]:
        future = self._pending.get(text)
        if future is None:
            future = asyncio.ensure_future(self.inner.aembed_query(text))
            self._pending[text] = future
            self.calls += 1
        try:
            return list(await future)
        except Exception:
            self._pending.pop(text, None)  # retried by the next caller
            raise


def _say(options: RagOptions, text: str) -> None:
    if options.progress is not None:
        options.progress(text)


async def _index(run: _Run) -> dict[str, Any]:
    from kartrix.context.indexers.pg_index import index_repo

    stats: dict[str, Any] = {}
    for name, root in run.roots.items():
        _say(run.options, f"Indexing fixture {name} …")
        start = time.perf_counter()
        result = await index_repo(root)
        stats[name] = {"seconds": round(time.perf_counter() - start, 2), "stats": str(result), "chunks": result.chunks}
    return stats


async def _retrieve_all(run: _Run, questions: list[GoldenQuestion]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(run.options.concurrency)
    rows: list[dict[str, Any]] = []

    async def one(q: GoldenQuestion, mode: str) -> None:
        retriever = get_retriever(mode)
        async with semaphore:
            start = time.perf_counter()
            try:
                chunks = await retriever(q.question, run.roots[q.repo], DEPTH)
                error = None
            except Exception as e:
                logger.warning("Retrieval failed", extra={"question": q.id, "mode": mode, "error": repr(e)})
                chunks, error = [], f"{type(e).__name__}: {e}"
            latency = (time.perf_counter() - start) * 1000
        run.retrieved[(q.id, mode)] = chunks
        top = chunks[: settings.retrieval.top_k]
        rows.append(
            {
                "question": q.id,
                "repo": q.repo,
                "kind": q.kind,
                "mode": mode,
                "latency_ms": round(latency, 1),
                "context_tokens": sum(estimate_tokens(format_chunk(c)) for c in top),
                "files": distinct_files(c["source"] for c in chunks)[:10],
                "scores": retrieval_scores([c["source"] for c in chunks], [c.get("name") for c in chunks], q.files, q.symbols),
                "error": error,
            }
        )  # fmt: skip

    await asyncio.gather(*(one(q, m) for q in questions for m in run.options.modes))
    order = {m: i for i, m in enumerate(run.options.modes)}
    ids = {q.id: i for i, q in enumerate(questions)}
    return sorted(rows, key=lambda r: (ids[r["question"]], order[r["mode"]]))


async def _answer_and_judge(
    run: _Run, questions: list[GoldenQuestion], mode: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from deepeval.test_case import LLMTestCase

    from kartrix.evals.judge import KartrixJudge, measure, rag_metrics

    answerer = KartrixJudge(role="main", concurrency=run.options.concurrency)
    judge = KartrixJudge(role="judge", concurrency=run.options.concurrency) if run.options.judge else None
    rows: list[dict[str, Any]] = []

    async def one(q: GoldenQuestion) -> None:
        chunks = run.retrieved.get((q.id, mode)) or []
        context = [format_chunk(c) for c in chunks[: settings.retrieval.top_k]]
        start = time.perf_counter()
        try:
            answer = await answerer.a_generate(
                ANSWER_PROMPT.format(question=q.question, context="\n\n---\n\n".join(context) or "(none)")
            )
            error = None
        except Exception as e:
            answer, error = "", f"{type(e).__name__}: {e}"
        row: dict[str, Any] = {
            "question": q.id,
            "repo": q.repo,
            "kind": q.kind,
            "mode": mode,
            "answer": answer,
            "answer_seconds": round(time.perf_counter() - start, 2),
            "error": error,
            "judge": {},
        }
        if judge is not None and not error:
            case = LLMTestCase(
                input=q.question,
                actual_output=answer,
                expected_output=q.answer,
                retrieval_context=context or ["(no context retrieved)"],
            )
            metrics = rag_metrics(judge)
            results = await asyncio.gather(*(measure(m, case) for m in metrics.values()))
            row["judge"] = dict(zip(metrics, results, strict=True))
        rows.append(row)  # fmt: skip

    await asyncio.gather(*(one(q) for q in questions))
    ids = {q.id: i for i, q in enumerate(questions)}
    rows.sort(key=lambda r: ids[r["question"]])
    usage = {"answer_model": answerer.usage.as_dict(), "judge_model": judge.usage.as_dict() if judge else None}
    return rows, usage


def summarise_retrieval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    keys = ["mrr", *(f"{m}@{k}" for m in ("hit", "recall", "precision", "ndcg") for k in K_VALUES)]
    keys += ["chunk_precision@5", "symbol_recall@5"]
    summary: dict[str, Any] = {}
    for mode, items in by_mode.items():
        latencies = [r["latency_ms"] for r in items if not r["error"]]
        summary[mode] = {
            "questions": len(items),
            "errors": sum(1 for r in items if r["error"]),
            **{k: _round(mean(r["scores"][k] for r in items)) for k in keys},
            "latency_ms_p50": _round(percentile(latencies, 0.5), 1),
            "latency_ms_p95": _round(percentile(latencies, 0.95), 1),
            "context_tokens_mean": _round(mean(r["context_tokens"] for r in items), 0),
            "by_repo": _breakdown(items, "repo"),
            "by_kind": _breakdown(items, "kind"),
        }
    return summary


def _breakdown(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in items:
        groups[r[key]].append(r)
    return {
        g: {
            "n": len(rs),
            "hit@5": _round(mean(r["scores"]["hit@5"] for r in rs)),
            "recall@5": _round(mean(r["scores"]["recall@5"] for r in rs)),
            "mrr": _round(mean(r["scores"]["mrr"] for r in rs)),
        }
        for g, rs in sorted(groups.items())
    }


def summarise_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from kartrix.evals.judge import RAG_JUDGE_METRICS

    out: dict[str, Any] = {
        "questions": len(rows),
        "answer_errors": sum(1 for r in rows if r["error"]),
        "answer_seconds_mean": _round(mean(r["answer_seconds"] for r in rows), 2),
    }
    for name in RAG_JUDGE_METRICS:
        scores = [r["judge"][name]["score"] for r in rows if name in r["judge"]]
        out[name] = _round(mean(scores))
        out[f"{name}_pass_rate"] = _round(
            mean(1.0 if s is not None and s >= 0.5 else 0.0 for s in scores if s is not None)
        )
        out[f"{name}_errors"] = sum(1 for r in rows if name in r["judge"] and r["judge"][name]["error"])
    return out


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


async def run_rag(options: RagOptions) -> dict[str, Any]:
    import kartrix.context.retrievers.pg_hybrid as pg_hybrid

    questions = load_golden(options.quick, options.ids)
    repos = load_repos()
    run = _Run(options)
    for name in sorted({q.repo for q in questions}):
        _say(options, f"Preparing fixture {name} …")
        run.roots[name] = await asyncio.to_thread(ensure_repo, repos[name], options.refresh_fixtures)

    needs_index = any(m in INDEX_MODES for m in (*options.modes, *options.answer_modes))
    index_stats = await _index(run) if needs_index else {}

    original = pg_hybrid.get_embedder  # wrap whatever the retriever uses (tests replace it)
    embedder = CachingEmbeddings(original()) if needs_index else None
    if embedder is not None:
        pg_hybrid.get_embedder = lambda: embedder  # type: ignore[assignment,return-value]
    try:
        modes = tuple(dict.fromkeys((*options.modes, *options.answer_modes)))
        run.options = RagOptions(**{**options.__dict__, "modes": modes})
        _say(options, f"Retrieving: {len(questions)} questions × {len(modes)} modes …")
        retrieval = await _retrieve_all(run, questions)
    finally:
        pg_hybrid.get_embedder = original

    answers: dict[str, Any] = {}
    for mode in options.answer_modes:
        _say(options, f"Answering and judging with {mode} context …")
        rows, usage = await _answer_and_judge(run, questions, mode)
        answers[mode] = {"summary": summarise_answers(rows), "usage": usage, "items": rows}

    measured = [r for r in retrieval if r["mode"] in options.modes]
    return {
        "questions": len(questions),
        "repos": {name: repos[name].model_dump() for name in run.roots},
        "index": index_stats,
        "query_embeddings": embedder.calls if embedder else 0,
        "retrieval": {"summary": summarise_retrieval(measured), "items": measured},
        "answers": answers,
    }
