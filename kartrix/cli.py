"""Command-line entry point.

- ``kartrix`` — start the interactive session in the current directory.
- ``kartrix stop`` — kill switch: stop every Kartrix run in progress on this machine
  (see :mod:`kartrix.security.kill_switch`). Imports almost nothing, so it works instantly.
- ``kartrix run --spec <file>`` — one headless run (no prompts; approvals from the spec), JSON report
  and exit code — see :mod:`kartrix.headless.runner`.
- ``kartrix eval rag|agent|all|calibrate|report`` — the agent and RAG eval suites, from a source checkout
  (see :mod:`kartrix.evals.cli`).
- ``kartrix trace [run]`` — the recent runs, or one run's timeline from its local trace
  (:mod:`kartrix.observability.tracing`).
- ``kartrix sandbox [status|check|reset]`` — which sandbox runs commands here (:mod:`kartrix.sandbox`),
  a live check of what it blocks, and (Windows) removing the folder access it was given.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("kartrix")
    except PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kartrix", description="Kartrix - security-first AI coding agent.")
    parser.add_argument("--version", action="version", version=f"kartrix {_version()}")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser(
        "stop",
        help="stop every Kartrix run in progress on this machine (kill switch)",
        description="Stops every Kartrix turn or /plan run in progress (in any terminal) at its next step; "
        "running commands are killed within a second. Kartrix itself keeps running.",
    )
    run_parser = commands.add_parser(
        "run",
        help="run one task headless (no prompts) from a spec file; prints a JSON report",
        description="Runs the spec's task in the current directory without asking anything: approvals come from the "
        "spec's approvals section, a plan is approved as proposed. Exit code: 0 completed, 1 failed, 2 stopped "
        "(budget / kill switch), 3 could not start.",
    )
    run_parser.add_argument("--spec", required=True, help="YAML run spec (see kartrix/headless/spec.py)")
    run_parser.add_argument("--report", help="write the JSON report here instead of stdout")
    run_parser.add_argument("--events", help="write every event as a JSON line to this file")
    run_parser.add_argument("--quiet", action="store_true", help="no progress on stderr")
    sandbox = commands.add_parser(
        "sandbox",
        help="show which sandbox runs commands here, check it, or reset it",
        description="status: the sandbox in use and why others aren't available. check: run a probe in it and show "
        "what it can and can't do. reset (Windows): remove every folder permission given to Kartrix's sandboxes.",
    )
    sandbox.add_argument("action", nargs="?", choices=["status", "check", "reset"], default="status")
    trace = commands.add_parser(
        "trace",
        help="list recent runs, or show one run's timeline (agents, model calls, tools, tokens)",
        description="Every run is traced to .kartrix/traces/ in the project. Without an argument: the recent runs. "
        "With a run id (or a prefix, or 'last'): its timeline, totals per agent and model, and the slowest steps.",
    )
    trace.add_argument("run", nargs="?", help="run id, prefix or 'last'")
    trace.add_argument("--json", action="store_true", help="print the raw JSON-lines trace")
    trace.add_argument("--stats", action="store_true", help="usage and cost over all traced runs, per model")
    from kartrix.evals.cli import add_parser as add_eval_parser

    add_eval_parser(commands)
    args = parser.parse_args(argv)

    if args.command == "run":
        from kartrix.headless.runner import run_headless

        return run_headless(args.spec, args.report, args.events, args.quiet)
    if args.command == "sandbox":
        return _sandbox(args.action)
    if args.command == "trace":
        from kartrix.observability.tracing import cli as trace_cli

        return trace_cli(args.run, args.json, args.stats)
    if args.command == "eval":
        from kartrix.evals.cli import run_eval

        return run_eval(args)

    if args.command == "stop":
        from kartrix.security.kill_switch import request_stop, stop_file

        when = request_stop()
        print(f"Stop requested at {datetime.fromtimestamp(when):%H:%M:%S} ({stop_file()}).")
        print("Every Kartrix run in progress stops at its next step; runs started from now on are not affected.")
        return 0

    from kartrix.main import run

    run()
    return 0


def _sandbox(action: str) -> int:
    from kartrix.sandbox.manager import status

    current = status()
    print(f"Sandbox: {current.describe()}")
    for line in current.tried if current.backend is not None else []:
        print(f"  (not used — {line})")
    if action == "status":
        return 0 if current.backend is not None else 1
    if action == "reset":
        if sys.platform != "win32":
            print("Nothing to reset: only the Windows sandbox changes folder permissions.")
        else:
            from kartrix.sandbox.windows import AppContainerBackend

            cleaned = AppContainerBackend().reset()
            print(f"Removed the sandbox's access to {len(cleaned)} path(s); it is given again on the next command.")
        return 0
    if current.backend is None:
        return 1
    from kartrix.sandbox.selfcheck import run_check

    results = run_check(current.backend)
    for r in results:
        got = "?" if r.allowed is None else ("allowed" if r.allowed else "blocked")
        mark = "ok  " if r.allowed == r.expected else ("note" if r.tolerated else "FAIL")
        print(f"  [{mark}] {r.name.replace('_', ' '):<16} {got}")
    failed = [r.name for r in results if not r.ok]
    print("Sandbox check passed." if not failed else f"Sandbox check FAILED: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
