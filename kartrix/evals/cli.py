"""``kartrix eval rag|agent|all|calibrate|report`` — run the eval suites and write results + an HTML report.

Examples::

    kartrix eval rag --quick --no-judge          # retrieval only, fast subset (everyday check)
    kartrix eval rag                             # all questions, every mode, answers judged
    kartrix eval agent --repeat 3 --judge        # every task three times: pass@1 and pass^3
    kartrix eval all --quick --compare           # fast subset of everything vs the committed baseline
    kartrix eval all --save-baseline             # record a new baseline (evals/baselines/*.json)
    kartrix eval report <results.json>           # re-render a report

Imports nothing heavy at module level: ``kartrix --version`` must stay instant.

Exit codes: 0 done · 1 a regression against the baseline (with ``--compare --fail-on-regression``)
· 3 could not run (datasets, configuration, services).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def add_parser(commands: Any) -> None:
    p = commands.add_parser(
        "eval",
        help="run the agent and RAG eval suites (from a Kartrix source checkout)",
        description="Measures Kartrix itself: retrieval and answer quality on golden questions (rag), coding tasks "
        "run headless and checked by commands (agent), the judge against hand labels (calibrate). Results and an "
        "HTML report go to .kartrix/evals/results/.",
    )
    p.add_argument("suite", choices=["rag", "agent", "all", "calibrate", "report"])
    p.add_argument("results", nargs="?", help="report: the results.json to render")
    p.add_argument("--quick", action="store_true", help="the small everyday subset")
    p.add_argument("--only", help="comma-separated question / task ids")
    p.add_argument("--modes", help="rag: retrieval modes (default dense,lexical,hybrid,search_first)")
    p.add_argument("--answer-modes", help="rag: modes whose context is answered and judged (default: retrieval.mode)")
    p.add_argument("--no-judge", action="store_true", help="rag: retrieval metrics only (no LLM calls but embeddings)")
    p.add_argument("--judge", action="store_true", help="agent: also score tool/argument/answer/plan with the judge")
    p.add_argument("--repeat", type=int, default=1, help="agent: runs per task (pass^k needs k >= 2)")
    p.add_argument("--jobs", type=int, default=1, help="agent: tasks run in parallel")
    p.add_argument("--concurrency", type=int, default=4, help="rag: parallel retrievals / model calls")
    p.add_argument("--keep", action="store_true", help="agent: keep the task workspaces")
    p.add_argument("--refresh-fixtures", action="store_true", help="rag: fetch the fixture repos again")
    p.add_argument("--compare", nargs="?", const="baseline", help="compare with the baseline (or a results.json)")
    p.add_argument("--fail-on-regression", action="store_true", help="exit 1 when --compare finds a regression")
    p.add_argument("--save-baseline", action="store_true", help="store these results as the new baseline")
    p.add_argument("--out", help="results folder (default .kartrix/evals/results/<time>-<suite>)")


def _progress(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


async def _run_suites(args: argparse.Namespace, suites: list[str]) -> dict[str, Any]:
    from kartrix.config import settings

    ids = [i.strip() for i in args.only.split(",")] if args.only else None
    out: dict[str, Any] = {}
    try:
        if "rag" in suites:
            from kartrix.evals.rag import RagOptions, run_rag
            from kartrix.evals.retrievers import DEFAULT_MODES

            modes = tuple(m.strip() for m in args.modes.split(",")) if args.modes else DEFAULT_MODES
            if args.no_judge:
                answer_modes: tuple[str, ...] = ()
            elif args.answer_modes:
                answer_modes = tuple(m.strip() for m in args.answer_modes.split(","))
            else:
                configured = settings.retrieval.mode
                answer_modes = ("lexical" if configured == "sparse" else configured,)
            out["rag"] = await run_rag(
                RagOptions(modes=modes, answer_modes=answer_modes, judge=not args.no_judge, quick=args.quick,
                           ids=ids if args.suite == "rag" else None, concurrency=args.concurrency,
                           refresh_fixtures=args.refresh_fixtures, progress=_progress)
            )  # fmt: skip
        if "calibrate" in suites:
            from kartrix.evals.judge import KartrixJudge, calibrate

            _progress("Calibrating the judge against the hand labels …")
            judge = KartrixJudge(concurrency=args.concurrency)
            out["calibrate"] = {**await calibrate(judge), "usage": judge.usage.as_dict()}
        if "agent" in suites:
            from kartrix.evals.agent import AgentOptions, run_agent

            out["agent"] = await run_agent(
                AgentOptions(quick=args.quick, ids=ids if args.suite == "agent" else None, repeat=max(1, args.repeat),
                             jobs=args.jobs, judge=args.judge, keep=args.keep, progress=_progress)
            )  # fmt: skip
    finally:
        from kartrix.db.engine import dispose_engine

        await dispose_engine()
    return out


def run_eval(args: argparse.Namespace) -> int:
    import warnings

    from kartrix.evals import store
    from kartrix.evals.compare import compare
    from kartrix.evals.datasets import DatasetError

    # Progress uses ✓ / ✗ / …: UTF-8 when redirected to a file, never a crash on a legacy console.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(**({"errors": "replace"} if stream.isatty() else {"encoding": "utf-8", "errors": "replace"}))
    # tree-sitter-languages 1.10 calls an API tree-sitter 0.21 deprecates (pinned until the parser work).
    warnings.filterwarnings("ignore", message=r"Language\(path, name\) is deprecated", category=FutureWarning)
    from kartrix.evals.report import render_html, text_summary

    if args.suite == "report":
        if not args.results:
            print("kartrix eval report: give the results.json to render", file=sys.stderr)
            return 3
        path = Path(args.results)
        target = path.with_name("report.html")
        target.write_text(render_html(store.load_json(path)), encoding="utf-8")
        print(target)
        return 0

    suites = ["rag", "calibrate", "agent"] if args.suite == "all" else [args.suite]
    if args.suite == "all" and args.quick:
        suites = ["rag", "agent"]  # calibration is a separate, occasional run
    started = time.time()
    try:
        data = asyncio.run(_run_suites(args, suites))
    except DatasetError as e:
        print(f"kartrix eval: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("kartrix eval: interrupted", file=sys.stderr)
        return 2
    except Exception as e:  # missing key, database down, …
        print(f"kartrix eval: could not run: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    result: dict[str, Any] = {
        "suite": args.suite,
        "subset": "quick" if args.quick else ("only" if args.only else "full"),
        "started_at": datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "versions": store.versions(suites),
        **data,
    }

    comparisons = []
    if args.compare:
        for suite in suites:
            if args.compare == "baseline":
                baseline = store.load_baseline(suite)
            else:
                baseline = store.load_json(args.compare)
            if baseline is None or suite not in baseline:
                _progress(f"No baseline for {suite} to compare with.")
                continue
            if baseline.get("subset") != result["subset"]:
                _progress(
                    f"Note: the {suite} baseline is the {baseline.get('subset')} set, this run the {result['subset']} set."
                )
            comparisons.append(compare(suite, result, baseline))
        result["comparisons"] = comparisons

    run_dir = store.new_run_dir(args.suite, args.out)
    store.write_json(run_dir / "results.json", result)
    (run_dir / "report.html").write_text(render_html(result), encoding="utf-8")
    if args.save_baseline:
        for suite in suites:
            keep = {k: result[k] for k in ("suite", "subset", "started_at", "duration_s", "versions")}
            store.write_json(store.baseline_path(suite), {**keep, "suite": suite, suite: result[suite]})
            _progress(f"Baseline saved: {store.baseline_path(suite)}")

    for line in text_summary(result):
        print(line)
    print(f"Results: {run_dir / 'results.json'}")
    print(f"Report:  {run_dir / 'report.html'}")
    regressed = any(c["regressions"] and c["comparable"] for c in comparisons)
    return 1 if regressed and args.fail_on_regression else 0
