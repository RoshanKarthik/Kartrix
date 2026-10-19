"""Command-line entry point.

- ``kartrix`` — start the interactive session in the current directory: through the Kartrix server after
  ``kartrix login``, else (or with ``--local``) in this process.
- ``kartrix serve`` — run the Kartrix server (the API for the CLI, web app and VS Code extension).
- ``kartrix login`` / ``logout`` / ``whoami`` — the account the CLI uses on a server.
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
import asyncio
import getpass
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
    parser.add_argument(
        "--local", action="store_true", help="run the agent in this process even when logged in to a server"
    )
    parser.add_argument("--model", help="catalog id of the model to use (see `kartrix models`; /model switches)")
    commands = parser.add_subparsers(dest="command", metavar="command")
    serve = commands.add_parser(
        "serve",
        help="run the Kartrix server (API for the CLI, web app and VS Code extension)",
        description="Serves the HTTP API with a live event feed per run. Each workspace's agent runs in its own "
        "worker process. Settings: the server section of config.yaml.",
    )
    serve.add_argument("--host", help="address to listen on (default: server.host, 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default: server.port, 8765)")
    login = commands.add_parser("login", help="sign in to a Kartrix server (or create an account with --signup)")
    login.add_argument("--server", help="server URL (default: server.url, http://127.0.0.1:8765)")
    login.add_argument("--email", help="account email (asked if left out)")
    login.add_argument("--signup", action="store_true", help="create the account first")
    commands.add_parser("logout", help="sign out of the Kartrix server and forget the token")
    commands.add_parser("whoami", help="show the server and account the CLI uses")
    commands.add_parser(
        "models",
        help="list the models you can pick, their tiers and prices",
        description="Shows the model catalog: from the Kartrix server after `kartrix login` (unless --local), "
        "else this install's config.",
    )
    worker = commands.add_parser("worker")  # started by the server, not listed in --help
    worker.add_argument("--session", required=True)
    worker.add_argument("--mcp", action="store_true")
    worker.add_argument("--model")
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

    if args.command == "serve":
        from kartrix.server.app import serve as run_server

        return run_server(args.host, args.port)
    if args.command == "worker":
        from kartrix.server.worker import run_worker

        return run_worker(args.session, args.mcp, args.model)
    if args.command == "models":
        return asyncio.run(_models(args.local))
    if args.command in ("login", "logout", "whoami"):
        secrets = _ask_login(args) if args.command == "login" else None
        if args.command == "login" and secrets is None:
            return 1
        return asyncio.run(_account(args, secrets))
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

    if not args.local:
        from kartrix.client.api import load_credentials

        creds = load_credentials()
        if creds is not None:
            from kartrix.client.repl import run_remote

            try:
                return asyncio.run(run_remote(creds, model=args.model))
            except KeyboardInterrupt:
                return 130

    from kartrix.main import run

    run(args.model)
    return 0


async def _models(local: bool) -> int:
    from rich.console import Console

    from kartrix.client.api import KartrixClient, load_credentials
    from kartrix.ui.model_list import local_catalog, render_catalog

    console = Console()
    creds = None if local else load_credentials()
    if creds is None:
        render_catalog(console, local_catalog(), "this install")
        return 0
    try:
        async with KartrixClient(creds.server, creds.token) as client:
            data = await client.models()
    except Exception as e:
        console.print(f"[red]Can't list the models of {creds.server}: {e}[/red]")
        return 1
    render_catalog(console, data, creds.server)
    return 0


def _ask_login(args: argparse.Namespace) -> tuple[str, str] | None:
    """Email and password from the terminal (before the event loop starts)."""
    email = args.email or input("Email: ").strip()
    password = getpass.getpass("Password: ")
    if args.signup and getpass.getpass("Password again: ") != password:
        print("The passwords don't match.")
        return None
    return email, password


async def _account(args: argparse.Namespace, secrets: tuple[str, str] | None = None) -> int:
    from kartrix.client.api import (
        ApiError,
        Credentials,
        KartrixClient,
        forget_credentials,
        load_credentials,
        save_credentials,
    )

    creds = load_credentials()
    if args.command == "whoami":
        if creds is None:
            print("Not logged in — the CLI runs the agent locally. `kartrix login` to use a server.")
            return 1
        try:
            async with KartrixClient(creds.server, creds.token) as client:
                me = await client.me()
        except ApiError as e:
            print(f"{creds.server}: {e.detail}")
            return 1
        except Exception as e:
            print(f"Can't reach {creds.server}: {e}")
            return 1
        print(f"{me['email']} on {creds.server}")
        return 0
    if args.command == "logout":
        if creds is not None:
            try:
                async with KartrixClient(creds.server, creds.token) as client:
                    await client.logout()
            except Exception as e:  # the token is forgotten locally either way
                print(f"The server could not revoke the token ({e}); it expires on its own.")
        print("Logged out." if forget_credentials() or creds else "Not logged in.")
        return 0

    from kartrix.config import settings

    server = (args.server or settings.server.url).rstrip("/")
    assert secrets is not None  # noqa: S101 — asked by main() for login
    email, password = secrets
    try:
        async with KartrixClient(server) as client:
            result = await (client.signup if args.signup else client.login)(email, password)
    except ApiError as e:
        print(f"{'Sign-up' if args.signup else 'Login'} failed: {e.detail}")
        return 1
    except Exception as e:
        print(f"Can't reach {server}: {e} — is `kartrix serve` running?")
        return 1
    path = save_credentials(Credentials(server, result["token"], result["user"]["email"]))
    print(f"Logged in to {server} as {result['user']['email']} (token saved in {path}).")
    print("`kartrix` now runs your requests on this server; `kartrix --local` runs them here.")
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
