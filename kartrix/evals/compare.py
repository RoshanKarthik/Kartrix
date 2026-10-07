"""``--compare``: did a change make Kartrix better or worse than the baseline?

Each tracked metric has a direction and a tolerance (absolute for rates and scores, relative for
time, tokens and cost, which are noisier); a change beyond the tolerance in the wrong direction is a
regression, in the right direction an improvement. Results measured on different datasets are flagged
as not comparable rather than compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Tracked:
    path: tuple[str, ...]  # inside the suite's result
    label: str
    higher_is_better: bool
    tolerance: float
    relative: bool = False  # tolerance is a fraction of the baseline value


def _retrieval(mode: str) -> list[Tracked]:
    base = ("retrieval", "summary", mode)
    return [
        Tracked((*base, "hit@5"), f"{mode} hit@5", True, 0.02),
        Tracked((*base, "recall@5"), f"{mode} recall@5", True, 0.02),
        Tracked((*base, "mrr"), f"{mode} MRR", True, 0.02),
        Tracked((*base, "ndcg@10"), f"{mode} nDCG@10", True, 0.02),
        Tracked((*base, "latency_ms_p50"), f"{mode} latency p50 (ms)", False, 0.5, relative=True),
    ]


def _answers(mode: str) -> list[Tracked]:
    base = ("answers", mode, "summary")
    return [
        Tracked((*base, name), f"{mode} {name.replace('_', ' ')}", True, 0.05)
        for name in ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall")
    ]


def tracked_metrics(suite: str, result: dict[str, Any]) -> list[Tracked]:
    if suite == "rag":
        modes = list(result.get("retrieval", {}).get("summary", {}))
        return [m for mode in modes for m in _retrieval(mode)] + [
            m for mode in result.get("answers", {}) for m in _answers(mode)
        ]
    if suite == "agent":
        s = ("summary",)
        repeat = result.get("summary", {}).get("repeat", 1)
        return [
            Tracked((*s, "pass@1"), "pass@1", True, 0.05),
            Tracked((*s, f"pass^{repeat}"), f"pass^{repeat}", True, 0.05),
            Tracked((*s, "safety_success"), "safety tasks passed", True, 0.0),
            Tracked((*s, "blocked_command_runs"), "blocked commands that ran", False, 0.0),
            Tracked((*s, "budget_respected_rate"), "budgets respected", True, 0.0),
            Tracked((*s, "tool_error_rate"), "tool error rate", False, 0.05),
            Tracked((*s, "invalid_args_rate"), "invalid argument rate", False, 0.02),
            Tracked((*s, "redundant_call_rate"), "redundant call rate", False, 0.05),
            Tracked((*s, "tokens_mean"), "tokens per run", False, 0.2, relative=True),
            Tracked((*s, "cost_usd_total"), "cost (USD)", False, 0.2, relative=True),
            Tracked((*s, "seconds_mean"), "time per run (s)", False, 0.25, relative=True),
            Tracked((*s, "startup_seconds_mean"), "startup (s)", False, 0.25, relative=True),
        ]
    if suite == "calibrate":
        return [
            Tracked(("summary", name, "agreement"), f"judge {name} agreement", True, 0.05)
            for name in ("faithfulness", "answer_relevancy")
        ]
    return []


def _get(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return float(cur) if isinstance(cur, int | float) and not isinstance(cur, bool) else None


Verdict = Literal["ok", "better", "regressed", "new", "missing"]


def compare(suite: str, current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Rows of (metric, baseline, current, delta, verdict) plus whether the datasets match."""
    cur_ds = (current.get("versions", {}).get("datasets") or {}).get(suite)
    base_ds = (baseline.get("versions", {}).get("datasets") or {}).get(suite)
    rows = []
    for t in tracked_metrics(suite, current.get(suite, {})):
        new = _get(current.get(suite, {}), t.path)
        old = _get(baseline.get(suite, {}), t.path)
        verdict: Verdict
        if new is None and old is None:
            continue
        if old is None:
            verdict, delta = "new", None
        elif new is None:
            verdict, delta = "missing", None
        else:
            delta = new - old
            allowed = t.tolerance * abs(old) if t.relative else t.tolerance
            worse = -delta if t.higher_is_better else delta
            verdict = "regressed" if worse > allowed + 1e-9 else ("better" if -worse > allowed + 1e-9 else "ok")
        rows.append({"metric": t.label, "baseline": old, "current": new, "delta": delta, "verdict": verdict})
    return {
        "suite": suite,
        "comparable": cur_ds == base_ds,
        "datasets": {"current": cur_ds, "baseline": base_ds},
        "baseline_versions": {k: baseline.get("versions", {}).get(k) for k in ("git_commit", "model", "prompts")},
        "rows": rows,
        "regressions": [r["metric"] for r in rows if r["verdict"] == "regressed"],
        "improvements": [r["metric"] for r in rows if r["verdict"] == "better"],
    }
