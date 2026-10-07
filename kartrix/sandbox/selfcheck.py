"""``kartrix sandbox check``: run a probe in the sandbox and show what it could and couldn't do.

The probe (this Python interpreter) runs in a throw-away workspace with a ``.env`` and a ``.git``
folder and tries to: write the workspace, read ``.env``, write ``.git``, write outside the
workspace, write its temp folder, and open a connection to the internet.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kartrix.sandbox.base import Backend, SandboxRun
from kartrix.security import workspace as ws_mod
from kartrix.security.environment import scrubbed_env
from kartrix.security.workspace import Workspace
from kartrix.tools.process_runner import run_launch

_PROBE = r"""
import os, socket
from pathlib import Path
def t(name, fn):
    try:
        fn(); print(name, "ok")
    except OSError:
        print(name, "blocked")
t("write_workspace", lambda: Path("out.txt").write_text("x"))
t("read_env", lambda: Path(".env").read_text())
t("write_git", lambda: Path(".git/hooks/pre-commit").write_text("x"))
t("write_outside", lambda: Path(os.environ["KARTRIX_OUTSIDE"]).write_text("x"))
t("write_temp", lambda: Path(os.environ["TEMP" if os.name == "nt" else "TMPDIR"], "probe.txt").write_text("x"))
t("network", lambda: socket.create_connection(("pypi.org", 443), timeout=5).close())
"""

# What a sandbox must allow (True) or block (False); Landlock can't protect the workspace's own files.
EXPECTED = {"write_workspace": True, "read_env": False, "write_git": False, "write_outside": False,
            "write_temp": True, "network": False}  # fmt: skip
_LANDLOCK_EXCEPTIONS = {"read_env", "write_git"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    allowed: bool | None  # None: the probe didn't report it
    expected: bool
    tolerated: bool = False  # a known limit of this backend

    @property
    def ok(self) -> bool:
        return self.allowed == self.expected or self.tolerated


def run_check(backend: Backend) -> list[CheckResult]:
    base = Path(tempfile.mkdtemp(prefix="kartrix-sandbox-check-"))
    root = base / "ws"
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".env").write_text("SECRET=probe\n", encoding="utf-8")
    outside = base / "outside.txt"
    previous = ws_mod._current
    try:
        ws_mod._current = Workspace.create(root)
        argv = [sys.executable, "-c", _PROBE]
        env = {**scrubbed_env(), "KARTRIX_OUTSIDE": str(outside)}
        result = run_launch(backend.prepare(SandboxRun(argv, argv, root, env, "off", root)), 60)
        seen = dict(line.split() for line in result.stdout.splitlines() if len(line.split()) == 2)
        out = []
        for name, expected in EXPECTED.items():
            allowed = None if name not in seen else seen[name] == "ok"
            if name == "write_outside" and allowed:  # bubblewrap: a private /tmp — did it reach the host?
                allowed = outside.exists()
            tolerated = backend.name == "Landlock" and name in _LANDLOCK_EXCEPTIONS
            out.append(CheckResult(name, allowed, expected, tolerated))
        return out
    finally:
        ws_mod._current = previous
        if sys.platform == "win32":
            from kartrix.sandbox.windows import AppContainerBackend

            if isinstance(backend, AppContainerBackend):
                backend.forget(root)
        shutil.rmtree(base, ignore_errors=True)
