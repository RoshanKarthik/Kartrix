"""Retrieval metrics and aggregation helpers (pure functions, no I/O).

Retrieval is scored **per file**: a retriever returns ranked chunks, which become a ranked list of
distinct files (first occurrence wins); a golden question lists the files that answer it. With
``ranked`` = that file list, ``relevant`` = the labelled files and cut-off ``k``:

- hit@k        1 if any relevant file is in the top k
- recall@k     share of the relevant files in the top k
- precision@k  share of the top k that is relevant (divided by k, so returning fewer files costs)
- MRR          1 / rank of the first relevant file (0 if none was returned)
- nDCG@k       binary gains, discounted by log2(rank + 1), normalised by the ideal ranking

Chunk precision@k (share of the first k *chunks* that come from a relevant file) is what the agent
actually reads — reported next to the file metrics.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

K_VALUES = (1, 3, 5, 10)


def distinct_files(sources: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for s in sources:
        seen.setdefault(normalise_path(s), None)
    return list(seen)


def normalise_path(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def hit_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(f in relevant for f in ranked[:k]) else 0.0


def recall_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, f in enumerate(ranked, start=1):
        if f in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, f in enumerate(ranked[:k], start=1) if f in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def chunk_precision_at(chunk_files: Sequence[str], relevant: set[str], k: int) -> float:
    top = [normalise_path(f) for f in chunk_files[:k]]
    return sum(f in relevant for f in top) / k


def symbol_recall_at(chunk_names: Sequence[str | None], symbols: Sequence[str], k: int) -> float | None:
    """Share of the expected symbols that name one of the top-k chunks (methods match ``Class.method``
    or ``method``). None when the question lists no symbols."""
    if not symbols:
        return None
    names = {n for n in chunk_names[:k] if n}
    short = {n.rsplit(".", 1)[-1] for n in names}
    found = sum(1 for s in symbols if s in names or s.rsplit(".", 1)[-1] in short)
    return found / len(symbols)


def retrieval_scores(
    chunk_files: Sequence[str], chunk_names: Sequence[str | None], relevant_files: Iterable[str], symbols: Sequence[str]
) -> dict[str, float | None]:
    """Every retrieval metric for one question."""
    relevant = {normalise_path(f) for f in relevant_files}
    ranked = distinct_files(chunk_files)
    scores: dict[str, float | None] = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in K_VALUES:
        scores[f"hit@{k}"] = hit_at(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at(ranked, relevant, k)
        scores[f"precision@{k}"] = precision_at(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at(ranked, relevant, k)
    scores["chunk_precision@5"] = chunk_precision_at(chunk_files, relevant, 5)
    scores["symbol_recall@5"] = symbol_recall_at(chunk_names, symbols, 5)
    return scores


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are not None (None if there are none)."""
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def pass_at_1(successes: Sequence[Sequence[bool]]) -> float | None:
    """Unbiased pass@1 over tasks: the mean success rate of each task's runs, averaged."""
    rates = [sum(runs) / len(runs) for runs in successes if runs]
    return statistics.fmean(rates) if rates else None


def pass_hat_k(successes: Sequence[Sequence[bool]], k: int) -> float | None:
    """pass^k (consistency): the chance that k independent runs of a task *all* succeed, averaged over
    tasks — estimated from n >= k runs with c successes as C(c, k) / C(n, k)."""
    values = [math.comb(sum(runs), k) / math.comb(len(runs), k) for runs in successes if len(runs) >= k]
    return statistics.fmean(values) if values else None


def cohen_kappa(labels: Sequence[bool], predictions: Sequence[bool]) -> float | None:
    """Agreement between the judge and the hand labels beyond chance (1 perfect, 0 chance level)."""
    n = len(labels)
    if n == 0 or n != len(predictions):
        return None
    observed = sum(a == b for a, b in zip(labels, predictions, strict=True)) / n
    p_yes = (sum(labels) / n) * (sum(predictions) / n)
    p_no = (1 - sum(labels) / n) * (1 - sum(predictions) / n)
    expected = p_yes + p_no
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
