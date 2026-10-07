"""Sandbox backends for Linux (bubblewrap, Landlock) and macOS (Seatbelt).

All three run the command through ``_helper.py`` (resource limits, Landlock, proxy bridge).

- **bubblewrap** — new mount, PID, IPC and (unless the network is "full") network namespace:
  the whole filesystem read-only, the workspace and state folder writable, protected paths
  re-mounted read-only or hidden, a private ``/tmp`` and ``/proc``. Installs reach the
  registries through the proxy's Unix socket, bridged to ``127.0.0.1`` inside.
- **Landlock** (no bubblewrap installed) — kernel access rules: everything readable except the
  credential folders in the home directory, writes only to the workspace and state folder,
  TCP blocked (kernel ABI 4+). Weaker: the workspace's own secrets stay readable and ``.git``
  writable — changes to protected paths are detected after the run and reported.
- **Seatbelt** (``sandbox-exec``) — a generated profile: writes only to the workspace (minus
  protected paths), state folder and ``/tmp``; credential folders and protected paths unreadable;
  keychain services blocked; network only to localhost (the proxy, local dev servers).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.sandbox import policy
from kartrix.sandbox.base import Backend, SandboxError, SandboxRun, SandboxUnavailable
from kartrix.sandbox.proxy import get_proxy
from kartrix.security.workspace import Workspace
from kartrix.tools.process_runner import Launch, ProcessResult

logger = get_logger(__name__)

HELPER = Path(__file__).with_name("_helper.py")
_MB = 1024 * 1024
_BRIDGE_PORT = 18080  # inside the sandbox's own network namespace, so it can't clash
_TRUSTED_BIN_DIRS = ("/usr/bin", "/bin", "/usr/local/bin")


def _workspace(run: SandboxRun) -> Workspace:
    return Workspace.create(run.workspace)


def _user_threads() -> int:
    """Processes + threads of this user right now (RLIMIT_NPROC counts all of them)."""
    uid = str(getattr(os, "getuid")())  # noqa: B009 — Linux only
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            status = Path(f"/proc/{entry}/status").read_text()
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        if fields.get("Uid", "").split()[:1] == [uid]:
            total += int(fields.get("Threads", "1").strip() or 1)
    return total


def rlimits() -> dict[str, int | None]:
    limits = settings.sandbox.limits
    out: dict[str, int | None] = {"core": 0}
    if limits.max_file_mb:
        out["fsize"] = limits.max_file_mb * _MB
    if sys.platform.startswith("linux"):
        if limits.memory_mb:
            out["data"] = limits.memory_mb * _MB
        if limits.max_processes:
            out["nproc"] = _user_threads() + limits.max_processes
    return out


def helper_command(config: dict[str, Any]) -> list[str]:
    return [sys.executable, "-I", "-S", str(HELPER), json.dumps(config)]


def _argv(run: SandboxRun) -> list[str]:
    if isinstance(run.args, str):  # pre-quoted batch command lines exist only on Windows
        raise SandboxError("internal error: a Windows command line reached a POSIX sandbox")
    return list(run.args)


def _probe(args: list[str]) -> str | None:
    """None if ``args`` runs fine, else why not."""
    try:
        proc = subprocess.run(args, capture_output=True, timeout=15, check=False)  # noqa: S603 — fixed probe
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    if proc.returncode != 0:
        return proc.stderr.decode(errors="replace").strip()[:300] or f"exit code {proc.returncode}"
    return None


# ── bubblewrap ────────────────────────────────────────────────────────


def find_bwrap() -> Path | None:
    for d in _TRUSTED_BIN_DIRS:  # never PATH: a planted bwrap would be the sandbox
        p = Path(d) / "bwrap"
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _hide(path: Path) -> list[str]:
    if path.is_dir():
        return ["--tmpfs", str(path), "--remount-ro", str(path)]
    return ["--ro-bind", "/dev/null", str(path)]


class BubblewrapBackend(Backend):
    name = "bubblewrap"
    registries_enforced = True

    def __init__(self, bwrap: Path) -> None:
        self.bwrap = bwrap

    @classmethod
    def detect(cls) -> BubblewrapBackend:
        bwrap = find_bwrap()
        if bwrap is None:
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed")
        true = shutil.which("true", path="/usr/bin:/bin") or "/bin/true"
        error = _probe([str(bwrap), "--ro-bind", "/", "/", "--unshare-net", "--unshare-pid", "--", true])
        if error:
            raise SandboxUnavailable(f"bubblewrap can't create namespaces here ({error})")
        return cls(bwrap)

    def prepare(self, run: SandboxRun) -> Launch:
        ws = _workspace(run)
        state = policy.state_dir(ws.root)
        prot = policy.protected_paths(ws)
        args = [str(self.bwrap), "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
                "--unshare-uts", "--unshare-cgroup-try"]  # fmt: skip
        if run.network != "full":
            args.append("--unshare-net")
        args += ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]  # noqa: S108 — a private /tmp
        for secret in policy.home_secrets(ws.root):
            args += _hide(secret)
        args += ["--bind", str(ws.root), str(ws.root)]
        for p in prot.readonly:
            args += ["--ro-bind", str(p), str(p)]
        for p in prot.hidden:
            args += _hide(p)
        for p in [state, *policy.extra_paths("write")]:
            if p.exists():
                args += ["--bind", str(p), str(p)]
        config: dict[str, Any] = {"argv": _argv(run), "rlimits": rlimits()}
        proxy_url = None
        if run.network == "registries":
            proxy = get_proxy()
            if proxy.unix_path is None:
                raise SandboxError("the registry proxy has no Unix socket")
            args += ["--bind", str(proxy.unix_path), str(proxy.unix_path)]
            config["bridge"] = {"port": _BRIDGE_PORT, "socket": str(proxy.unix_path)}
            proxy_url = proxy.url.replace(f":{proxy.port}", f":{_BRIDGE_PORT}")
        args += ["--chdir", str(run.cwd), "--", *helper_command(config)]
        env = policy.sandbox_env(run.env, state, run.network, proxy_url)
        return Launch(args, run.cwd, env)


# ── Landlock ──────────────────────────────────────────────────────────


def _allowed_tree(root: Path, denied: list[Path]) -> list[str]:
    """Paths that together cover ``root`` except ``denied`` (Landlock rules only add access)."""
    denied_s = {os.path.normpath(str(d)) for d in denied}

    def visit(path: str) -> list[str]:
        if path in denied_s:
            return []
        prefix = path.rstrip(os.sep) + os.sep
        if not any(d.startswith(prefix) for d in denied_s):
            return [path]
        out: list[str] = []
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return []
        for name in entries:
            out += visit(os.path.join(path, name))
        return out

    return visit(os.path.normpath(str(root)))


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """(mtime, size) of every file under the protected paths (git objects excluded)."""
    snap: dict[str, tuple[int, int]] = {}
    for top in paths:
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d != "objects"]
            for name in [*filenames, "."]:
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p, follow_symlinks=False)
                except OSError:
                    continue
                snap[os.path.normpath(p)] = (st.st_mtime_ns, st.st_size)
            if len(snap) > 50_000:
                return snap
        if top.is_file():
            st = top.stat()
            snap[str(top)] = (st.st_mtime_ns, st.st_size)
    return snap


class LandlockBackend(Backend):
    name = "Landlock"
    registries_enforced = False
    loopback = False

    def __init__(self, abi: int) -> None:
        self.abi = abi

    @classmethod
    def detect(cls) -> LandlockBackend:
        from kartrix.sandbox._helper import landlock_abi

        abi = landlock_abi()
        if abi < 1:
            raise SandboxUnavailable("Landlock is not enabled in this kernel")
        if abi < 4:
            raise SandboxUnavailable(f"Landlock ABI {abi} can't block the network (needs Linux 6.7+)")
        return cls(abi)

    def describe(self) -> str:
        return (
            f"Landlock (ABI {self.abi}) — writes limited to the workspace, no network; installs need approval. "
            "Install bubblewrap for the full sandbox (protected workspace files, registry-only installs)"
        )

    def prepare(self, run: SandboxRun) -> Launch:
        ws = _workspace(run)
        state = policy.state_dir(ws.root)
        prot = policy.protected_paths(ws)
        secrets = policy.home_secrets(ws.root)
        read = _allowed_tree(Path("/"), secrets)
        write = [str(p) for p in [ws.root, state, *policy.extra_paths("write"), Path("/dev")]]
        config = {
            "argv": _argv(run),
            "rlimits": rlimits(),
            "landlock": {"read": read, "write": write, "network_off": run.network != "full"},
        }
        env = policy.sandbox_env(run.env, state, run.network)
        watched = [p for p in prot.readonly + prot.hidden if p.exists()]
        before = _snapshot(watched)

        def after(result: ProcessResult) -> ProcessResult:
            now = _snapshot(watched)
            changed = sorted(p for p in before.keys() | now.keys() if before.get(p) != now.get(p))
            if not changed:
                return result
            rels = [os.path.relpath(p, ws.root) for p in changed[:10]]
            more = f" and {len(changed) - 10} more" if len(changed) > 10 else ""
            logger.warning("Sandboxed command changed protected files", extra={"paths": rels})
            note = (
                f"\nWARNING: this command changed protected files ({', '.join(rels)}{more}). "
                "Tell the user — they should review these changes (e.g. git hooks) before using git."
            )
            return replace(result, stderr=result.stderr + note)

        return Launch(helper_command(config), run.cwd, env, after=after)


# ── Seatbelt (macOS) ──────────────────────────────────────────────────

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_KEYCHAIN_SERVICES = ("com.apple.SecurityServer", "com.apple.security.agent", "com.apple.securityd",
                      "com.apple.secd", "com.apple.trustd.agent")  # fmt: skip


def seatbelt_profile(
    write: list[Path], readonly: list[Path], hidden: list[Path], network: str
) -> tuple[str, dict[str, str]]:
    """SBPL profile + its parameters (paths go in as ``-D`` parameters, never into the text)."""
    params: dict[str, str] = {}

    def refs(prefix: str, paths: list[Path]) -> str:
        out = []
        for i, p in enumerate(paths):
            key = f"{prefix}{i}"
            params[key] = os.path.realpath(p)
            out.append(f'(subpath (param "{key}"))')
        return " ".join(out)

    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    lines.append(
        f'(allow file-write* {refs("W", write)} (literal "/dev/null") (literal "/dev/zero") '
        '(literal "/dev/dtracehelper") (regex #"^/dev/tty") (regex #"^/dev/fd/") (subpath "/private/tmp"))'
    )
    if readonly:
        lines.append(f"(deny file-write* {refs('R', readonly)})")
    if hidden:
        lines.append(f"(deny file-read* file-write* {refs('H', hidden)})")
    lines.append("(deny mach-lookup " + " ".join(f'(global-name "{s}")' for s in _KEYCHAIN_SERVICES) + ")")
    if network != "full":
        lines += [
            "(deny network*)",
            '(allow network* (local ip "localhost:*"))',
            '(allow network-outbound (remote ip "localhost:*"))',
        ]
    return "\n".join(lines) + "\n", params


class SeatbeltBackend(Backend):
    name = "Seatbelt"
    registries_enforced = True

    @classmethod
    def detect(cls) -> SeatbeltBackend:
        if not _SANDBOX_EXEC.is_file():
            raise SandboxUnavailable("sandbox-exec is missing")
        error = _probe([str(_SANDBOX_EXEC), "-p", "(version 1)(allow default)", "/usr/bin/true"])
        if error:
            raise SandboxUnavailable(f"sandbox-exec doesn't work here ({error})")
        return cls()

    def prepare(self, run: SandboxRun) -> Launch:
        ws = _workspace(run)
        state = policy.state_dir(ws.root)
        prot = policy.protected_paths(ws)
        hidden = policy.home_secrets(ws.root) + prot.hidden
        write = [ws.root, state, *policy.extra_paths("write")]
        profile, params = seatbelt_profile(write, prot.readonly, hidden, run.network)
        args = [str(_SANDBOX_EXEC), "-p", profile]
        for key, value in params.items():
            args += ["-D", f"{key}={value}"]
        args += helper_command({"argv": _argv(run), "rlimits": rlimits()})
        proxy_url = get_proxy().url if run.network == "registries" else None
        env = policy.sandbox_env(run.env, state, run.network, proxy_url)
        return Launch(args, run.cwd, env)
