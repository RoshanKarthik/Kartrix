"""Command-line entry point.

- ``kartrix`` — start the interactive session in the current directory.
- ``kartrix stop`` — kill switch: stop every Kartrix run in progress on this machine
  (see :mod:`kartrix.security.kill_switch`). Imports almost nothing, so it works instantly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kartrix", description="Kartrix - security-first AI coding agent.")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser(
        "stop",
        help="stop every Kartrix run in progress on this machine (kill switch)",
        description="Stops every Kartrix turn or /plan run in progress (in any terminal) at its next step; "
        "running commands are killed within a second. Kartrix itself keeps running.",
    )
    args = parser.parse_args(argv)

    if args.command == "stop":
        from kartrix.security.kill_switch import request_stop, stop_file

        when = request_stop()
        print(f"Stop requested at {datetime.fromtimestamp(when):%H:%M:%S} ({stop_file()}).")
        print("Every Kartrix run in progress stops at its next step; runs started from now on are not affected.")
        return 0

    from kartrix.main import run

    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
