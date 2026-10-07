"""The eval report: one self-contained HTML file (tables + inline SVG charts, light and dark), plus a short
plain-text summary for the terminal. No dependencies, no network: it opens anywhere and can be shared.
"""

from __future__ import annotations

import html
from typing import Any

# Categorical slots in fixed order (validated reference palette), light / dark.
_SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500")]
_MODE_LABEL = {"dense": "Dense", "lexical": "Lexical", "hybrid": "Hybrid (RRF)", "search_first": "Search-first"}

_CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f3f2ef; --grid:#e4e3df; --text:#0b0b0b;
  --text-2:#52514e; --muted:#7a7974; --good:#008300; --bad:#c62f2e; --warn:#a86b00;
  %(light)s }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --surface:#1a1a19; --surface-2:#242422; --grid:#383835; --text:#ffffff; --text-2:#c3c2b7; --muted:#99988f;
  --good:#4cb84c; --bad:#e66767; --warn:#e0a52e; %(dark)s } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --surface-2:#242422; --grid:#383835;
  --text:#ffffff; --text-2:#c3c2b7; --muted:#99988f; --good:#4cb84c; --bad:#e66767; --warn:#e0a52e; %(dark)s }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:40px 0 8px; } h3 { font-size:15px; margin:24px 0 8px; }
p.sub { color:var(--text-2); margin:0 0 16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:12px; margin:16px 0; }
.tile { background:var(--surface-2); border-radius:8px; padding:12px 14px; }
.tile .label { color:var(--text-2); font-size:13px; } .tile .value { font-size:24px; font-weight:600; }
.tile .note { color:var(--muted); font-size:12px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { color:var(--text-2); font-weight:600; font-size:13px; } td.num, th.num { text-align:right; }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:8px 0; color:var(--text-2); font-size:13px; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:6px; background:var(--c); vertical-align:-1px; }
svg text { fill:var(--text-2); font-size:12px; } svg .val { fill:var(--text); }
.ok { color:var(--good); } .bad { color:var(--bad); } .warn { color:var(--warn); }
details { margin:8px 0; } summary { cursor:pointer; color:var(--text-2); }
code { font-size:12px; }
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _tile(label: str, value: str, note: str = "") -> str:
    return f'<div class="tile"><div class="label">{_esc(label)}</div><div class="value">{value}</div>' + (
        f'<div class="note">{_esc(note)}</div></div>' if note else "</div>"
    )


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{_esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def grouped_bars(groups: list[str], series: list[str], values: dict[str, list[float | None]], title: str) -> str:
    """Horizontal grouped bars on a 0..1 scale: one group per metric, one bar per series."""
    bar, gap, group_gap, label_w, plot_w = 12, 2, 14, 120, 520
    group_h = len(series) * (bar + gap) - gap
    height = len(groups) * (group_h + group_gap) + 24
    parts = [f'<svg viewBox="0 0 {label_w + plot_w + 60} {height}" width="100%" role="img" aria-label="{_esc(title)}">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + t * plot_w
        parts.append(f'<line x1="{x}" x2="{x}" y1="0" y2="{height - 20}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{height - 6}" text-anchor="middle">{t:.2g}</text>')
    for gi, group in enumerate(groups):
        y0 = gi * (group_h + group_gap)
        parts.append(f'<text x="{label_w - 8}" y="{y0 + group_h / 2 + 4}" text-anchor="end">{_esc(group)}</text>')
        for si, name in enumerate(series):
            v = values[name][gi]
            if v is None:
                continue
            y = y0 + si * (bar + gap)
            w = max(1.0, v * plot_w)
            r = min(4.0, w / 2)
            x = label_w
            path = f"M{x},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{bar - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r} z"
            tip = f"{name} · {group}: {v:.3f}"
            parts.append(f'<path d="{path}" fill="var(--s{si + 1})"><title>{_esc(tip)}</title></path>')
        best = max((values[n][gi] or 0) for n in series)
        parts.append(
            f'<text class="val" x="{label_w + best * plot_w + 6}" y="{y0 + group_h / 2 + 4}">{best:.2f}</text>'
        )
    parts.append("</svg>")
    legend = (
        '<div class="legend">'
        + "".join(f'<span style="--c:var(--s{i + 1})">{_esc(n)}</span>' for i, n in enumerate(series))
        + "</div>"
        if len(series) > 1
        else ""
    )
    return legend + "".join(parts)


def _rag_section(rag: dict[str, Any]) -> str:
    summary = rag["retrieval"]["summary"]
    modes = list(summary)
    out = [f'<h2>RAG — retrieval</h2><p class="sub">{rag["questions"]} golden questions over '
           f'{len(rag["repos"])} fixture repos; metrics per file (top-k distinct files).</p>']  # fmt: skip
    best = max(modes, key=lambda m: summary[m].get("mrr") or 0) if modes else None
    if best:
        s = summary[best]
        out.append('<div class="tiles">')
        out.append(_tile("Best mode (MRR)", _esc(_MODE_LABEL.get(best, best))))
        out.append(_tile("Hit@5", _pct(s.get("hit@5")), _MODE_LABEL.get(best, best)))
        out.append(_tile("Recall@5", _pct(s.get("recall@5")), _MODE_LABEL.get(best, best)))
        out.append(_tile("MRR", _fmt(s.get("mrr"), 2), _MODE_LABEL.get(best, best)))
        out.append("</div>")
    metrics = ["hit@1", "hit@5", "recall@5", "recall@10", "precision@5", "mrr", "ndcg@10", "chunk_precision@5"]
    labels = [_MODE_LABEL.get(m, m) for m in modes]
    out.append(
        grouped_bars(
            metrics, labels, {lab: [summary[m].get(k) for k in metrics] for lab, m in zip(labels, modes, strict=True)},
            "Retrieval metrics per mode",
        )
    )  # fmt: skip
    rows = [
        [_esc(_MODE_LABEL.get(m, m)), *(_fmt(summary[m].get(k)) for k in metrics),
         _fmt(summary[m].get("latency_ms_p50"), 0), _fmt(summary[m].get("context_tokens_mean"), 0)]
        for m in modes
    ]  # fmt: skip
    out.append(_table(["Mode", *metrics, "p50 ms", "ctx tokens"], rows, set(range(1, len(metrics) + 3))))
    for m in modes:
        by_repo = summary[m]["by_repo"]
        by_kind = summary[m]["by_kind"]
        out.append(f"<details><summary>{_esc(_MODE_LABEL.get(m, m))} by repo and question kind</summary>")
        out.append(_table(["Repo", "n", "hit@5", "recall@5", "MRR"],
                          [[_esc(k), v["n"], _fmt(v["hit@5"]), _fmt(v["recall@5"]), _fmt(v["mrr"])] for k, v in by_repo.items()],
                          {1, 2, 3, 4}))  # fmt: skip
        out.append(_table(["Kind", "n", "hit@5", "recall@5", "MRR"],
                          [[_esc(k), v["n"], _fmt(v["hit@5"]), _fmt(v["recall@5"]), _fmt(v["mrr"])] for k, v in by_kind.items()],
                          {1, 2, 3, 4}))  # fmt: skip
        out.append("</details>")
    misses = [r for r in rag["retrieval"]["items"] if r["mode"] == best and not r["scores"]["hit@5"]]
    if misses:
        out.append(f"<details><summary>{len(misses)} questions the best mode missed in its top 5</summary>")
        out.append(_table(["Question", "Repo", "Returned (top 5 files)"],
                          [[_esc(r["question"]), _esc(r["repo"]), _esc(", ".join(r["files"][:5]))] for r in misses]))  # fmt: skip
        out.append("</details>")
    for mode, ans in rag.get("answers", {}).items():
        s = ans["summary"]
        names = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]
        out.append(f"<h2>RAG — answers ({_esc(_MODE_LABEL.get(mode, mode))} context, LLM judge)</h2>")
        out.append('<div class="tiles">' + "".join(
            _tile(n.replace("_", " ").capitalize(), _fmt(s.get(n), 2), f"pass rate {_pct(s.get(n + '_pass_rate'))}")
            for n in names) + "</div>")  # fmt: skip
        judge_usage = (ans.get("usage") or {}).get("judge_model") or {}
        out.append(f'<p class="sub">{s["questions"]} answers · {s["answer_errors"]} answer errors · judge: '
                   f'{_fmt(judge_usage.get("calls"))} calls, {_fmt(judge_usage.get("failures"))} failures.</p>')  # fmt: skip
        rows = [
            [_esc(r["question"]), *(_fmt((r["judge"].get(n) or {}).get("score"), 2) for n in names)]
            for r in ans["items"]
        ]
        out.append(
            "<details><summary>Scores per question</summary>"
            + _table(["Question", *names], rows, {1, 2, 3, 4})
            + "</details>"
        )
    return "".join(out)


def _agent_section(agent: dict[str, Any]) -> str:
    s = agent["summary"]
    k = s["repeat"]
    out = [f'<h2>Agent tasks</h2><p class="sub">{s["tasks"]} tasks × {k} run(s), each headless in a fresh '
           f"workspace, checked by commands.</p>"]  # fmt: skip
    out.append('<div class="tiles">')
    out.append(_tile("Task success (pass@1)", _pct(s.get("pass@1"))))
    if k > 1:
        out.append(_tile(f"Consistency (pass^{k})", _pct(s.get(f"pass^{k}"))))
    out.append(_tile("Safety tasks passed", _pct(s.get("safety_success"))))
    out.append(_tile("Tool calls per run", _fmt(s.get("tool_calls_mean"), 1)))
    out.append(_tile("Tokens per run", _fmt(s.get("tokens_mean"))))
    out.append(_tile("Time per run", f"{_fmt(s.get('seconds_mean'), 0)} s", f"p95 {_fmt(s.get('seconds_p95'), 0)} s"))
    out.append(_tile("Startup", f"{_fmt(s.get('startup_seconds_mean'), 2)} s", "before indexing"))
    out.append("</div>")
    cats = s["by_category"]
    out.append("<h3>Success by category</h3>")
    out.append(
        grouped_bars(list(cats), ["success"], {"success": [v["success"] for v in cats.values()]}, "Success by category")
    )
    traj = [
        ["Invalid tool arguments", _pct(s.get("invalid_args_rate")), "of all tool calls"],
        ["Redundant calls", _pct(s.get("redundant_call_rate")), "same call again with no write in between"],
        ["Tool errors", _pct(s.get("tool_error_rate")), "of all tool calls"],
        ["Recovery", _pct(s.get("recovery_rate")), "runs with a tool error that still succeeded"],
        ["Expected tools used", _pct(s.get("expected_tool_recall")), "share of the task's expected tools"],
        ["Model calls per run", _fmt(s.get("model_calls_mean"), 1), ""],
        ["Approvals asked per run", _fmt(s.get("human_interventions_mean"), 2), "each one interrupts a person"],
        ["Approval asked when expected", _pct(s.get("approval_asked_rate")), ""],
        ["Blocked commands that ran", _fmt(s.get("blocked_command_runs")), "must be 0"],
        ["Budgets respected", _pct(s.get("budget_respected_rate")), ""],
        [
            "Cost (total)",
            f"${_fmt(s.get('cost_usd_total'), 4)}" + ("" if s.get("cost_known") else " (lower bound)"),
            "",
        ],
        *([f"Judge: {n.replace('_', ' ')}", _fmt(v, 2), "DeepEval"] for n, v in s.get("judge", {}).items()),
    ]
    out.append("<h3>Trajectory, efficiency and safety</h3>" + _table(["Metric", "Value", "Note"], traj, {1}))
    rows = []
    for r in agent["items"]:
        mark = '<span class="ok">✓</span>' if r["success"] else '<span class="bad">✗</span>'
        usage = r.get("usage") or {}
        rows.append([
            mark, _esc(r["task"]) + (f" #{r['attempt']}" if k > 1 else ""), _esc(r["category"]), _esc(r["status"]),
            _fmt(r["trajectory"]["total"]), _fmt(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
            _fmt(usage.get("seconds"), 0), _esc("; ".join(r["failures"])[:300]),
        ])  # fmt: skip
    out.append(
        "<h3>Runs</h3>"
        + _table(["", "Task", "Category", "Status", "Tool calls", "Tokens", "s", "Why it failed"], rows, {4, 5, 6})
    )
    return "".join(out)


def _calibration_section(cal: dict[str, Any]) -> str:
    rows = [
        [_esc(name.replace("_", " ")), v["items"], _pct(v["agreement"]), _fmt(v["kappa"], 2), v["false_pass"], v["false_fail"]]
        for name, v in cal["summary"].items()
    ]  # fmt: skip
    return ("<h2>Judge calibration</h2><p class=\"sub\">The judge against hand labels (threshold "
            f"{cal['threshold']}). Kappa ≥ 0.6 is substantial agreement.</p>"
            + _table(["Metric", "Items", "Agreement", "Cohen's κ", "False pass", "False fail"], rows, {1, 2, 3, 4, 5}))  # fmt: skip


def _comparison_section(comparisons: list[dict[str, Any]]) -> str:
    out = []
    for c in comparisons:
        note = "" if c["comparable"] else ' <span class="warn">(datasets differ — not comparable)</span>'
        out.append(f"<h2>Compared with the baseline — {_esc(c['suite'])}{note}</h2>")
        base = c.get("baseline_versions") or {}
        out.append(f'<p class="sub">Baseline: commit {_esc((base.get("git_commit") or "?")[:10])}, '
                   f'model {_esc(base.get("model"))}, prompts {_esc(base.get("prompts"))}.</p>')  # fmt: skip
        cls = {"regressed": "bad", "better": "ok"}
        rows = [
            [_esc(r["metric"]), _fmt(r["baseline"]), _fmt(r["current"]),
             "—" if r["delta"] is None else f"{r['delta']:+.3g}",
             f'<span class="{cls.get(r["verdict"], "")}">{_esc(r["verdict"])}</span>']
            for r in c["rows"]
        ]  # fmt: skip
        out.append(_table(["Metric", "Baseline", "Now", "Δ", ""], rows, {1, 2, 3}))
    return "".join(out)


def render_html(result: dict[str, Any]) -> str:
    v = result.get("versions", {})
    light = " ".join(f"--s{i + 1}:{c[0]};" for i, c in enumerate(_SERIES))
    dark = " ".join(f"--s{i + 1}:{c[1]};" for i, c in enumerate(_SERIES))
    body = [
        "<h1>Kartrix eval report</h1>",
        f'<p class="sub">{_esc(result.get("started_at"))} · {_esc(v.get("model"))} · judge {_esc(v.get("judge_model"))} · '
        f'commit {_esc((v.get("git_commit") or "?")[:10])}{" (dirty)" if v.get("git_dirty") else ""} · '
        f'sandbox {_esc(v.get("sandbox"))}</p>',
    ]  # fmt: skip
    if result.get("comparisons"):
        body.append(_comparison_section(result["comparisons"]))
    if "rag" in result:
        body.append(_rag_section(result["rag"]))
    if "calibrate" in result:
        body.append(_calibration_section(result["calibrate"]))
    if "agent" in result:
        body.append(_agent_section(result["agent"]))
    ver_rows = [[_esc(k), f"<code>{_esc(val)}</code>"] for k, val in v.items()]
    body.append("<h2>Versions</h2>" + _table(["", ""], ver_rows))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Kartrix eval report</title><style>{_CSS % {'light': light, 'dark': dark}}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>"
    )


def text_summary(result: dict[str, Any]) -> list[str]:
    """A few lines for the terminal."""
    lines = []
    if "rag" in result:
        for mode, s in result["rag"]["retrieval"]["summary"].items():
            lines.append(
                f"RAG {mode:<13} hit@5 {_pct(s['hit@5']):>4}  recall@5 {_pct(s['recall@5']):>4}  "
                f"MRR {_fmt(s['mrr'], 2):>4}  nDCG@10 {_fmt(s['ndcg@10'], 2):>4}  p50 {_fmt(s['latency_ms_p50'], 0)} ms"
            )
        for mode, ans in result["rag"].get("answers", {}).items():
            s = ans["summary"]
            lines.append(
                f"RAG answers ({mode}): faithfulness {_fmt(s.get('faithfulness'), 2)}  relevancy "
                f"{_fmt(s.get('answer_relevancy'), 2)}  ctx precision {_fmt(s.get('contextual_precision'), 2)}  "
                f"ctx recall {_fmt(s.get('contextual_recall'), 2)}"
            )
    if "calibrate" in result:
        for name, s in result["calibrate"]["summary"].items():
            lines.append(f"Judge {name}: agreement {_pct(s['agreement'])}, kappa {_fmt(s['kappa'], 2)}")
    if "agent" in result:
        s = result["agent"]["summary"]
        k = s["repeat"]
        lines.append(
            f"Agent: pass@1 {_pct(s['pass@1'])}" + (f", pass^{k} {_pct(s.get(f'pass^{k}'))}" if k > 1 else "")
            + f", safety {_pct(s['safety_success'])}, {_fmt(s['tool_calls_mean'], 1)} tool calls, "
            f"{_fmt(s['tokens_mean'])} tokens, {_fmt(s['seconds_mean'], 0)} s per run"
        )  # fmt: skip
    for c in result.get("comparisons", []):
        if not c["comparable"]:
            lines.append(f"Compare {c['suite']}: datasets differ from the baseline — not comparable")
        else:
            lines.append(
                f"Compare {c['suite']}: {len(c['regressions'])} regressed"
                + (f" ({', '.join(c['regressions'])})" if c["regressions"] else "")
                + f", {len(c['improvements'])} improved"
            )
    return lines
