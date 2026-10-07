"""Do the workspace's tests catch planted bugs? (checks for "write tests for X" tasks)

    python .eval/mutation_check.py --mutant src/pricing.ts=.eval/mutants/a.ts [--mutant …] -- node --test

1. The test command must pass on the workspace as it is (the agent's tests are green).
2. For each mutant, a copy of the workspace with that one file replaced must make the test command FAIL.

Exit 0 only if both hold; prints which mutants survived. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SKIP = shutil.ignore_patterns(".git", ".eval", ".kartrix", "node_modules", "__pycache__", ".pytest_cache")


def run(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)  # noqa: S603
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    print(f"$ {' '.join(cmd)}  (in {cwd.name}) -> exit {proc.returncode}")
    print("\n".join("    " + line for line in tail))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant", action="append", required=True, help="target=replacement (workspace-relative)")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    root = Path.cwd()
    if run(cmd, root) != 0:
        print("FAIL: the tests do not pass on the unchanged code")
        return 1
    survived = []
    for spec in args.mutant:
        target, replacement = spec.split("=", 1)
        with tempfile.TemporaryDirectory(prefix="mutant-", dir=root / ".eval") as tmp:
            copy = Path(tmp) / "ws"
            shutil.copytree(root, copy, ignore=_SKIP)
            shutil.copyfile(root / replacement, copy / target)
            if run(cmd, copy) == 0:
                survived.append(replacement)
    if survived:
        print(f"FAIL: the tests did not catch {len(survived)} planted bug(s): {', '.join(survived)}")
        return 1
    print(f"OK: the tests caught all {len(args.mutant)} planted bug(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
