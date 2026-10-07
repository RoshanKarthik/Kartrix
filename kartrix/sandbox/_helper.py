"""Runs inside the sandbox (Linux, macOS) just before the command — standard library only.

Started as ``python -I -S _helper.py <config-json>`` (never imported as part of Kartrix there),
it applies what can only be applied from inside, then runs the command:

1. resource limits (``setrlimit``): file size, data size, processes, no core dumps;
2. Landlock (Linux without bubblewrap): read-only everywhere except the allowed paths,
   writes only to the allowed ones, and TCP blocked when the network is off;
3. with a network-namespaced sandbox, a bridge from ``127.0.0.1:<port>`` to the registry
   proxy's Unix socket, kept running while the command runs.

Without a bridge the helper replaces itself with the command (``execv``), so signals and the
exit code are the command's own. Kartrix imports this module only for :func:`landlock_abi`.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
from typing import Any

# ── Landlock ──────────────────────────────────────────────────────────

_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446  # same number on every Linux architecture
_PR_SET_NO_NEW_PRIVS = 38
_RULE_PATH_BENEATH = 1
_CREATE_RULESET_VERSION = 1

_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_REFER, _TRUNCATE, _IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_FILE_ONLY = _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE | _IOCTL_DEV  # rights valid on a file rule
_READ = _EXECUTE | _READ_FILE | _READ_DIR
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1


_LINUX = sys.platform.startswith("linux")  # a variable: type-checks the Linux code on every OS


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi() -> int:
    """Landlock ABI version of the running kernel (0 = unavailable)."""
    if not _LINUX:
        return 0
    try:
        version = _libc().syscall(_SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_RULESET_VERSION))
    except (OSError, AttributeError):
        return 0
    return max(int(version), 0)


def _handled_fs(abi: int) -> int:
    rights = (1 << 13) - 1  # ABI 1: execute … make_sym
    if abi >= 2:
        rights |= _REFER
    if abi >= 3:
        rights |= _TRUNCATE
    if abi >= 5:
        rights |= _IOCTL_DEV
    return rights


def apply_landlock(read: list[str], write: list[str], network_off: bool) -> None:
    abi = landlock_abi()
    if abi < 1:
        raise OSError("Landlock is not available")
    libc = _libc()
    handled = _handled_fs(abi)
    attr = _RulesetAttr(handled, (_NET_BIND_TCP | _NET_CONNECT_TCP) if network_off and abi >= 4 else 0)
    size = ctypes.sizeof(_RulesetAttr) if abi >= 4 else 8
    ruleset = libc.syscall(_SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for paths, rights in ((read, _READ), (write, handled)):
            for path in paths:
                try:
                    fd = os.open(path, getattr(os, "O_PATH", 0) | getattr(os, "O_CLOEXEC", 0))
                except OSError:
                    continue  # vanished or unreachable: nothing to allow
                try:
                    is_dir = os.path.isdir(path)
                    rule = _PathBeneath((rights if is_dir else rights & _FILE_ONLY) & handled, fd)
                    if libc.syscall(_SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), 0) < 0:
                        raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
                finally:
                    os.close(fd)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS) failed")
        if libc.syscall(_SYS_RESTRICT, ruleset, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)


# ── resource limits ───────────────────────────────────────────────────


def apply_rlimits(limits: dict[str, int | None]) -> None:
    resource: Any = importlib.import_module("resource")
    names = {"fsize": "RLIMIT_FSIZE", "data": "RLIMIT_DATA", "nproc": "RLIMIT_NPROC", "core": "RLIMIT_CORE"}
    for key, name in names.items():
        value = limits.get(key)
        if value is None or not hasattr(resource, name):
            continue
        which = getattr(resource, name)
        _soft, hard = resource.getrlimit(which)
        new = value if hard == resource.RLIM_INFINITY else min(value, hard)
        try:
            resource.setrlimit(which, (new, new))
        except (ValueError, OSError):
            pass  # best effort: some platforms (macOS) refuse some limits


# ── proxy bridge (network namespace → registry proxy) ────────────────


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def start_bridge(port: int, unix_path: str) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(64)

    def serve() -> None:
        while True:
            client, _ = server.accept()
            upstream = socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM)  # noqa: B009 — POSIX only
            try:
                upstream.connect(unix_path)
            except OSError:
                client.close()
                continue
            threading.Thread(target=_pump, args=(client, upstream), daemon=True).start()
            threading.Thread(target=_pump, args=(upstream, client), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────


def main(config: dict[str, Any]) -> int:
    argv: list[str] = config["argv"]
    if config.get("rlimits"):
        apply_rlimits(config["rlimits"])
    landlock = config.get("landlock")
    if landlock:
        apply_landlock(landlock["read"], landlock["write"], landlock["network_off"])
    bridge = config.get("bridge")
    if not bridge:
        os.execv(argv[0], argv)  # noqa: S606 — the command the policy allowed, now confined
    start_bridge(bridge["port"], bridge["socket"])
    child = subprocess.Popen(argv)  # noqa: S603 — the command the policy allowed, now confined
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(sig, lambda s, _f: child.send_signal(s))
    code = child.wait()
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    try:
        sys.exit(main(json.loads(sys.argv[1])))
    except OSError as e:
        print(f"kartrix sandbox: {e}", file=sys.stderr)
        sys.exit(126)
