"""Windows sandbox: AppContainer (no admin rights needed).

An AppContainer process runs with its own SID and can open only what an ACL grants to that SID
(or to "ALL APPLICATION PACKAGES" — Windows, Program Files). It has no network unless given the
``internetClient`` capability, and it can't reach servers on the host's loopback.

- One AppContainer profile per workspace (``kartrix.<hash>``), so a command in one project can't
  touch another project.
- Its SID is granted Modify on the workspace and on the state folder, and Read & execute on the
  toolchain the command uses (its folder, a venv's base interpreter, ``sandbox.extra_read``).
  The folders above the workspace and the toolchain get "list / read attributes" on that folder
  only (nothing below it inherits it): tools stat their parents (pytest looks for ``conftest.py``
  above the project, uv's launcher resolves its own path), and without it they fail with "access
  denied". Contents stay unreadable; folders the user may not change (a drive root) are skipped.
  Protected workspace paths stop inheriting the workspace grant (deny entries for an AppContainer
  SID aren't honoured, so the grant has to be absent): ``deny_read`` paths are then out of reach,
  ``deny_write`` paths get Read & execute back. Refreshed before every run, so new ``.env`` files
  are covered.
  Every grant is recorded in the user data folder; ``kartrix sandbox reset`` removes them all.
- Two Python details that break inside an AppContainer are worked around for sandboxed runs only:
  a venv's console-script launchers (``pytest.exe`` made by uv or pip) fail to resolve their own
  path, so they run as ``python.exe <launcher>`` (the launcher is a zip app); and Python 3.12.4+
  turns ``mkdir(mode=0o700)`` (``tempfile.mkdtemp``, pytest's cache and ``tmp_path``) into an
  owner-only ACL the container's SID can't open, so a ``sitecustomize`` shim on ``PYTHONPATH``
  creates those folders with the inherited ACL instead (any existing ``sitecustomize`` still runs).
- Node.js can't start child processes in an AppContainer (libuv's named pipes are refused and it
  retries forever; inherited handles fail with ENOENT), so ``node --test``, ``npm test``, install
  scripts and dev servers would hang. Node toolchain commands therefore run unsandboxed under the
  normal approval rules (``cannot_run``), and the approval prompt and audit log say so.
- Network: off, or — for an approved install or network command — ``internetClient`` (any host:
  without admin rights the network can't be limited to the registries on Windows).
- The process starts suspended, joins a job object with memory and process limits (see
  :mod:`kartrix.tools.process_runner`), and inherits only its stdin/stdout/stderr handles.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import random
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Iterator
from ctypes import wintypes
from pathlib import Path
from typing import Any, cast

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.paths import user_data_dir
from kartrix.sandbox import policy
from kartrix.sandbox.base import Backend, SandboxError, SandboxRun, SandboxUnavailable
from kartrix.security.workspace import Workspace
from kartrix.tools.process_runner import JobLimits, Launch

logger = get_logger(__name__)

_INTERNET_CLIENT = "S-1-15-3-1"
_ALREADY_EXISTS = -2147024713  # HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS) as a signed 32-bit value

# Access masks
_READ_EXECUTE = 0x1200A9  # FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
_MODIFY = 0x1301BF  # read, write, execute, delete
_TRAVERSE = 0x1000A1  # FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE
_SE_DACL_AUTO_INHERITED, _SE_DACL_PROTECTED = 0x0400, 0x1000
_GRANT, _REVOKE = 1, 4  # ACCESS_MODE
_INHERIT = 3  # SUB_CONTAINERS_AND_OBJECTS_INHERIT
_SE_FILE_OBJECT = 1
_DACL = 0x4

_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_STARTF_USESTDHANDLES = 0x00000100
_ATTR_HANDLE_LIST = 0x00020002
_ATTR_SECURITY_CAPABILITIES = 0x00020009

if sys.platform == "win32":
    _userenv = ctypes.WinDLL("userenv")
    _advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _win_error = ctypes.WinError
    _last_error = ctypes.get_last_error
    import msvcrt as _msvcrt
else:  # imported on other platforms for the cross-platform helpers; the Win32 calls never run there
    _userenv = _advapi = _k32 = _win_error = _last_error = _msvcrt = cast(Any, None)


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _SecurityCapabilities(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(_SidAndAttributes)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _Trustee(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),  # 0 = TRUSTEE_IS_SID
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),  # the SID
    ]


class _ExplicitAccess(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _Trustee),
    ]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


def _check_hr(hr: int, what: str) -> None:
    if hr != 0:
        raise OSError(f"{what} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")


class Sid:
    """An owned SID (freed on close)."""

    def __init__(self, ptr: ctypes.c_void_p, free: str) -> None:
        self.ptr = ptr
        self._free = free

    @classmethod
    def from_string(cls, text: str) -> Sid:
        ptr = ctypes.c_void_p()
        if not _advapi.ConvertStringSidToSidW(ctypes.c_wchar_p(text), ctypes.byref(ptr)):
            raise _win_error(_last_error())
        return cls(ptr, "local")

    def __str__(self) -> str:
        text = ctypes.c_wchar_p()
        if not _advapi.ConvertSidToStringSidW(self.ptr, ctypes.byref(text)):
            raise _win_error(_last_error())
        try:
            return str(text.value)
        finally:
            _k32.LocalFree(text)

    def close(self) -> None:
        if self.ptr:
            if self._free == "local":
                _k32.LocalFree(self.ptr)
            else:
                _advapi.FreeSid(self.ptr)
            self.ptr = ctypes.c_void_p()


def container_name(root: Path) -> str:
    return f"kartrix.{policy.workspace_key(root)}"


def ensure_profile(name: str, attempts: int = 6) -> Sid:
    """Create the AppContainer profile (or find the existing one) and return its SID.

    Two Kartrix processes creating the same profile at the same moment (two terminals, parallel
    eval jobs) make one call fail with E_UNEXPECTED; it is retried, and once the other process has
    created the profile the existing one is used."""
    for attempt in range(attempts):
        sid = ctypes.c_void_p()
        hr = ctypes.c_int32(
            _userenv.CreateAppContainerProfile(
                ctypes.c_wchar_p(name),
                ctypes.c_wchar_p("Kartrix sandbox"),
                ctypes.c_wchar_p("Commands Kartrix runs for one workspace"),
                None,
                wintypes.DWORD(0),
                ctypes.byref(sid),
            )
        ).value
        if hr >= 0:
            return Sid(sid, "free")
        if hr == _ALREADY_EXISTS:
            _check_hr(
                _userenv.DeriveAppContainerSidFromAppContainerName(ctypes.c_wchar_p(name), ctypes.byref(sid)),
                "DeriveAppContainerSid",
            )
            return Sid(sid, "free")
        if attempt + 1 < attempts:
            logger.info("AppContainer profile creation failed; retrying", extra={"hresult": hex(hr & 0xFFFFFFFF)})
            time.sleep(0.1 * (attempt + 1) + random.random() * 0.1)  # noqa: S311 — jitter, not security
    _check_hr(hr, "CreateAppContainerProfile")
    raise AssertionError("unreachable")  # _check_hr raised


def delete_profile(name: str) -> None:
    _userenv.DeleteAppContainerProfile(ctypes.c_wchar_p(name))


# ── ACLs ──────────────────────────────────────────────────────────────


def set_access(path: Path, sid: Sid, mask: int, mode: int, inherit: bool = True) -> None:
    """Add (or with ``_REVOKE`` remove) an ACE for ``sid`` on ``path``; inheritable entries
    propagate to everything below it."""
    old = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    err = _advapi.GetNamedSecurityInfoW(
        ctypes.c_wchar_p(str(path)), _SE_FILE_OBJECT, _DACL, None, None, ctypes.byref(old), None, ctypes.byref(sd)
    )
    if err:
        raise _win_error(err)
    try:
        ea = _ExplicitAccess(mask, mode, _INHERIT if inherit and path.is_dir() else 0, _Trustee(None, 0, 0, 0, sid.ptr))
        new = ctypes.c_void_p()
        err = _advapi.SetEntriesInAclW(1, ctypes.byref(ea), old, ctypes.byref(new))
        if err:
            raise _win_error(err)
        try:
            err = _advapi.SetNamedSecurityInfoW(
                ctypes.c_wchar_p(str(path)), _SE_FILE_OBJECT, _DACL, None, None, new, None
            )
            if err:
                raise _win_error(err)
        finally:
            _k32.LocalFree(new)
    finally:
        _k32.LocalFree(sd)


def set_access_here(path: Path, sid: Sid, mask: int, mode: int) -> None:
    """Add (or remove) a non-inheritable ACE on ``path`` alone. ``SetFileSecurityW`` writes this one
    security descriptor; ``SetNamedSecurityInfoW`` would walk everything below the folder (slow for a
    home directory, and needless: nothing changes there)."""
    old = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    err = _advapi.GetNamedSecurityInfoW(
        ctypes.c_wchar_p(str(path)), _SE_FILE_OBJECT, _DACL, None, None, ctypes.byref(old), None, ctypes.byref(sd)
    )
    if err:
        raise _win_error(err)
    try:
        control, revision = wintypes.WORD(), wintypes.DWORD()
        if not _advapi.GetSecurityDescriptorControl(sd, ctypes.byref(control), ctypes.byref(revision)):
            raise _win_error(_last_error())
        ea = _ExplicitAccess(mask, mode, 0, _Trustee(None, 0, 0, 0, sid.ptr))
        new = ctypes.c_void_p()
        err = _advapi.SetEntriesInAclW(1, ctypes.byref(ea), old, ctypes.byref(new))
        if err:
            raise _win_error(err)
        try:
            absolute = ctypes.create_string_buffer(64)  # SECURITY_DESCRIPTOR (40 bytes on x64)
            flags = _SE_DACL_AUTO_INHERITED | _SE_DACL_PROTECTED
            ok = (
                _advapi.InitializeSecurityDescriptor(absolute, 1)
                and _advapi.SetSecurityDescriptorDacl(absolute, True, new, False)
                and _advapi.SetSecurityDescriptorControl(absolute, flags, control.value & flags)
                and _advapi.SetFileSecurityW(ctypes.c_wchar_p(str(path)), _DACL, absolute)
            )
            if not ok:
                raise _win_error(_last_error())
        finally:
            _k32.LocalFree(new)
    finally:
        _k32.LocalFree(sd)


def parent_dirs(paths: list[Path]) -> list[Path]:
    """The folders above ``paths`` (outside the system folders, which are readable already)."""
    system = _system_dirs()
    out: list[Path] = []
    for path in paths:
        for parent in Path(os.path.realpath(path)).parents:
            if parent not in out and not any((os.path.normcase(str(parent)) + os.sep).startswith(s) for s in system):
                out.append(parent)
    return out


_thread_lock = threading.RLock()


@contextlib.contextmanager
def acl_lock(timeout: float = 120.0) -> Iterator[None]:
    """Held while a sandbox changes ACLs, across every Kartrix process of this user. Changing an ACL is a
    read-modify-write of the whole list: two processes granting their containers access to the same
    folder (a shared venv, a parent folder) at the same moment lost one entry, and that container could
    no longer start Python ("No pyvenv.cfg file") while the grant store said it had access."""

    with _thread_lock:
        path = user_data_dir() / "sandbox_grants.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as fh:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fh.seek(0)
                    _msvcrt.locking(fh.fileno(), _msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                _msvcrt.locking(fh.fileno(), _msvcrt.LK_UNLCK, 1)


class GrantStore:
    """Every ACE Kartrix added, per container — so ``kartrix sandbox reset`` can remove them. Reloaded
    from disk under :func:`acl_lock` before it is used, so processes don't overwrite each other's records."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or user_data_dir() / "sandbox_grants.json"
        self._lock = threading.Lock()
        self.data: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def key(self, path: Path) -> str:
        return os.path.normcase(str(path))

    def has(self, container: str, path: Path, kind: str) -> bool:
        entry = self.data.get(container, {}).get(self.key(path))
        if entry is None or entry.get("kind") != kind:
            return False
        try:  # a deleted and re-created file has a new file id and no deny entry yet
            return bool(entry.get("id") == path.stat().st_ino)
        except OSError:
            return False

    def add(self, container: str, path: Path, kind: str) -> None:
        with self._lock:
            self.data.setdefault(container, {})[self.key(path)] = {
                "path": str(path), "kind": kind, "id": path.stat().st_ino
            }  # fmt: skip
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    def remove_container(self, container: str) -> dict[str, Any]:
        with self._lock:
            entries = self.data.pop(container, {})
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
            return entries


_MASKS = {"modify": (_MODIFY, _GRANT), "read": (_READ_EXECUTE, _GRANT)}
_ICACLS = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icacls.exe"


def _icacls(path: Path, *args: str) -> None:
    proc = subprocess.run(  # noqa: S603 — fixed system binary
        [str(_ICACLS), str(path), *args, "/Q"], capture_output=True, timeout=120, check=False
    )
    if proc.returncode != 0:
        raise OSError(f"icacls {' '.join(args)} failed for {path}: {proc.stdout.decode(errors='replace').strip()}")


def protect(path: Path, sid: Sid, readable: bool) -> None:
    """Cut ``path`` off from the workspace grant: inherited entries become explicit copies,
    then every entry for ``sid`` is removed; ``readable`` gives Read & execute back."""
    _icacls(path, "/inheritance:d")
    _icacls(path, "/remove", f"*{sid}")
    if readable:
        _icacls(path, "/grant", f"*{sid}:{'(OI)(CI)' if path.is_dir() else ''}(RX)")


def unprotect(path: Path, sid: Sid) -> None:
    _icacls(path, "/inheritance:e")
    _icacls(path, "/remove", f"*{sid}")


def _system_dirs() -> list[str]:
    dirs = [os.environ.get(v, "") for v in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    return [os.path.normcase(d) + os.sep for d in dirs if d]


def toolchain_dirs(exe: Path) -> list[Path]:
    """Folders a command needs to read that the AppContainer can't by default."""
    out: list[Path] = []
    system = _system_dirs()
    real = Path(os.path.realpath(exe))
    folder = real.parent
    out.append(folder.parent if folder.name.lower() in ("scripts", "bin") else folder)
    for cfg in (folder / "pyvenv.cfg", folder.parent / "pyvenv.cfg"):  # a venv: its base interpreter too
        try:
            for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() == "home" and value.strip():
                    out.append(Path(value.strip()))
        except OSError:
            continue
    return [p for p in out if p.exists() and not any((os.path.normcase(str(p)) + os.sep).startswith(s) for s in system)]


# ── Python inside the container ───────────────────────────────────────

SHIM = '''"""Written by Kartrix's Windows sandbox (kartrix/sandbox/windows.py) for sandboxed commands only.

Python 3.12.4+ gives folders made with mode 0o700 (tempfile.mkdtemp, pytest's cache and tmp_path)
an owner-only ACL, which a process in an AppContainer can't open. Inside the sandbox those folders
get the ACL of their parent instead (the sandbox's own temp folder or the workspace).
"""
import importlib.machinery
import importlib.util
import os
import sys

if sys.platform == "win32":
    _mkdir = os.mkdir

    def _sandbox_mkdir(path, mode=0o777, *, dir_fd=None):
        return _mkdir(path, 0o777 if mode == 0o700 else mode, dir_fd=dir_fd)

    os.mkdir = _sandbox_mkdir

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here]
)
if _spec is not None and _spec.loader is not None:  # the environment's own sitecustomize, if any
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
'''


def python_shim(state: Path) -> Path:
    """The folder with the ``sitecustomize`` shim (rewritten when it differs)."""
    folder = state / "pyshim"
    target = folder / "sitecustomize.py"
    try:
        current = target.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != SHIM:
        folder.mkdir(parents=True, exist_ok=True)
        target.write_text(SHIM, encoding="utf-8")
    return folder


def unwrap_launcher(args: list[str]) -> list[str]:
    """`<venv>/Scripts/tool.exe ...` → `<venv>/Scripts/python.exe <tool.exe> ...` for console-script
    launchers (an .exe with a zip app appended, next to the venv's python.exe)."""
    if not args:
        return args
    exe = Path(args[0])
    python = exe.with_name("python.exe")
    if exe.suffix.lower() != ".exe" or exe.name.lower() in ("python.exe", "pythonw.exe") or not python.is_file():
        return args
    try:
        if not zipfile.is_zipfile(exe):
            return args
    except OSError:
        return args
    return [str(python), str(exe), *args[1:]]


NODE_PROGRAMS = frozenset({
    "node", "npm", "npx", "pnpm", "pnpx", "yarn", "yarnpkg", "corepack", "bun", "bunx", "deno", "tsc", "tsx",
    "ts-node", "vite", "vitest", "jest", "mocha", "next", "nuxt", "astro", "eslint", "prettier", "playwright",
    "webpack", "rollup", "esbuild", "turbo", "nx", "nodemon",
})  # fmt: skip
NODE_REASON = "Node.js can't start child processes inside an AppContainer"


def runs_on_node(argv: list[str], exe: Path | None) -> bool:
    """A Node.js toolchain command: a known Node program, anything from node_modules, or an npm-style
    ``.cmd``/``.ps1`` shim that starts node."""
    name = Path(argv[0]).name.lower() if argv else ""
    for suffix in (".exe", ".cmd", ".bat", ".ps1"):
        name = name.removesuffix(suffix)
    if name in NODE_PROGRAMS:
        return True
    if exe is None:
        return False
    if "node_modules" in (part.lower() for part in exe.parts):
        return True
    if exe.suffix.lower() in (".cmd", ".bat", ".ps1"):
        try:
            return "node" in exe.read_text(encoding="utf-8", errors="replace")[:4000].lower()
        except OSError:
            return False
    return False


# ── process creation ──────────────────────────────────────────────────


def _env_block(env: dict[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    items = sorted(env.items(), key=lambda kv: kv[0].upper())
    text = "".join(f"{k}={v}\0" for k, v in items if k and "=" not in k and "\0" not in k + v) + "\0"
    return ctypes.create_unicode_buffer(text, len(text))


class AppContainerPopen(subprocess.Popen[bytes]):
    """``Popen`` whose child starts suspended inside an AppContainer; ``resume()`` starts it."""

    def __init__(self, *args: Any, container_sid: Sid, capabilities: list[Sid], **kwargs: Any) -> None:
        self._container_sid = container_sid
        self._capabilities = capabilities
        self._thread_handle: int | None = None
        super().__init__(*args, **kwargs)

    def resume(self) -> None:
        if self._thread_handle is not None:
            _k32.ResumeThread(wintypes.HANDLE(self._thread_handle))
            _k32.CloseHandle(wintypes.HANDLE(self._thread_handle))
            self._thread_handle = None

    def _execute_child(
        self, args: Any, executable: Any, preexec_fn: Any, close_fds: Any, pass_fds: Any, cwd: Any, env: Any,
        startupinfo: Any, creationflags: int, shell: bool, p2cread: Any, p2cwrite: Any, c2pread: Any,
        c2pwrite: Any, errread: Any, errwrite: Any, *_unused: Any,
    ) -> None:  # fmt: skip
        cmdline = args if isinstance(args, str) else subprocess.list2cmdline(args)
        handles = list(dict.fromkeys(int(h) for h in (p2cread, c2pwrite, errwrite) if h not in (-1, None)))
        try:
            pi = self._spawn(cmdline, cwd, env, creationflags, p2cread, c2pwrite, errwrite, handles)
        finally:
            self._close_pipe_fds(p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite)  # type: ignore[attr-defined]
        self._child_created = True
        self._handle = subprocess.Handle(pi.hProcess)  # type: ignore[attr-defined]
        self.pid = pi.dwProcessId
        self._thread_handle = pi.hThread

    def _spawn(self, cmdline: str, cwd: Any, env: Any, flags: int, stdin: Any, stdout: Any, stderr: Any,
               handles: list[int]) -> _ProcessInformation:  # fmt: skip
        caps = (_SidAndAttributes * max(len(self._capabilities), 1))()
        for i, cap in enumerate(self._capabilities):
            caps[i] = _SidAndAttributes(cap.ptr, 0x4)  # SE_GROUP_ENABLED
        sc = _SecurityCapabilities(
            self._container_sid.ptr, caps if self._capabilities else None, len(self._capabilities), 0
        )
        handle_array = (wintypes.HANDLE * len(handles))(*handles)

        size = ctypes.c_size_t()
        _k32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        attrs = ctypes.create_string_buffer(size.value)
        if not _k32.InitializeProcThreadAttributeList(attrs, 2, 0, ctypes.byref(size)):
            raise _win_error(_last_error())
        try:
            for attr, value, length in (
                (_ATTR_SECURITY_CAPABILITIES, ctypes.byref(sc), ctypes.sizeof(sc)),
                (_ATTR_HANDLE_LIST, handle_array, ctypes.sizeof(handle_array)),
            ):
                if not _k32.UpdateProcThreadAttribute(
                    attrs, 0, ctypes.c_size_t(attr), value, ctypes.c_size_t(length), None, None
                ):
                    raise _win_error(_last_error())
            si = _StartupInfoEx()
            si.StartupInfo.cb = ctypes.sizeof(_StartupInfoEx)
            si.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
            si.StartupInfo.hStdInput = int(stdin)
            si.StartupInfo.hStdOutput = int(stdout)
            si.StartupInfo.hStdError = int(stderr)
            si.lpAttributeList = ctypes.cast(attrs, ctypes.c_void_p)
            pi = _ProcessInformation()
            buf = ctypes.create_unicode_buffer(cmdline)
            env_block = _env_block(dict(env)) if env is not None else None
            ok = _k32.CreateProcessW(
                None, buf, None, None, True,
                flags | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_UNICODE_ENVIRONMENT | _CREATE_SUSPENDED | _CREATE_NO_WINDOW,
                env_block, None if cwd is None else ctypes.c_wchar_p(os.fsdecode(cwd)),
                ctypes.byref(si), ctypes.byref(pi),
            )  # fmt: skip
            if not ok:
                raise _win_error(_last_error())
            return pi
        finally:
            _k32.DeleteProcThreadAttributeList(attrs)


def _bind_prototypes() -> None:
    _userenv.CreateAppContainerProfile.restype = ctypes.c_long
    _userenv.CreateAppContainerProfile.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                                   ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]  # fmt: skip
    _userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
    _userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    _userenv.DeleteAppContainerProfile.restype = ctypes.c_long
    _userenv.DeleteAppContainerProfile.argtypes = [ctypes.c_wchar_p]
    _advapi.ConvertStringSidToSidW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    _advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
    _advapi.FreeSid.argtypes = [ctypes.c_void_p]
    _advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.GetNamedSecurityInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]  # fmt: skip
    _advapi.SetEntriesInAclW.restype = wintypes.DWORD
    _advapi.SetEntriesInAclW.argtypes = [wintypes.ULONG, ctypes.POINTER(_ExplicitAccess), ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_void_p)]  # fmt: skip
    _advapi.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                                                     ctypes.POINTER(wintypes.DWORD)]  # fmt: skip
    _advapi.InitializeSecurityDescriptor.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _advapi.SetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, wintypes.BOOL, ctypes.c_void_p, wintypes.BOOL]
    _advapi.SetSecurityDescriptorControl.argtypes = [ctypes.c_void_p, wintypes.WORD, wintypes.WORD]
    _advapi.SetFileSecurityW.argtypes = [ctypes.c_wchar_p, wintypes.DWORD, ctypes.c_void_p]
    _advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.SetNamedSecurityInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]  # fmt: skip
    _k32.LocalFree.argtypes = [ctypes.c_void_p]
    _k32.LocalFree.restype = ctypes.c_void_p
    _k32.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                                       ctypes.POINTER(ctypes.c_size_t)]  # fmt: skip
    _k32.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t, ctypes.c_void_p,
                                               ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]  # fmt: skip
    _k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    _k32.CreateProcessW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p,
                                    wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, ctypes.c_wchar_p,
                                    ctypes.c_void_p, ctypes.c_void_p]  # fmt: skip
    _k32.ResumeThread.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]


# ── backend ───────────────────────────────────────────────────────────


class AppContainerBackend(Backend):
    name = "AppContainer"
    registries_enforced = False
    loopback = False

    def __init__(self) -> None:
        self.grants = GrantStore()
        self._sids: dict[str, Sid] = {}
        self._lock = threading.Lock()
        self._no_traverse: set[tuple[str, Path]] = set()  # parents we may not change (tried once per process)

    @classmethod
    def detect(cls) -> AppContainerBackend:
        if sys.platform != "win32":
            raise SandboxUnavailable("AppContainer exists only on Windows")
        else:  # an explicit branch, so type-checking for other platforms skips it instead of flagging it
            if sys.getwindowsversion().build < 9200:
                raise SandboxUnavailable("AppContainer needs Windows 8 or later")
            try:
                _bind_prototypes()
                ensure_profile("kartrix.probe")
            except (OSError, AttributeError) as e:
                raise SandboxUnavailable(f"AppContainer profiles can't be created ({e})") from e
            return cls()

    def describe(self) -> str:
        return "AppContainer — writes limited to the workspace, network off; installs need approval (then: internet)"

    def cannot_run(self, argv: list[str], exe: Path | None) -> str | None:
        return NODE_REASON if runs_on_node(argv, exe) else None

    def _sid(self, name: str) -> Sid:
        with self._lock:
            if name not in self._sids:
                self._sids[name] = ensure_profile(name)
            return self._sids[name]

    def _grant(self, name: str, sid: Sid, path: Path, kind: str) -> None:
        if not path.exists() or self.grants.has(name, path, kind):
            return
        logger.info("Sandbox access granted", extra={"container": name, "path": str(path), "kind": kind})
        if kind == "traverse":
            set_access_here(path, sid, _TRAVERSE, _GRANT)
        elif kind in _MASKS:
            mask, mode = _MASKS[kind]
            set_access(path, sid, mask, mode)
        else:
            protect(path, sid, readable=kind == "deny_write")
        self.grants.add(name, path, kind)

    def _grant_all(self, name: str, sid: Sid, ws: Workspace, state: Path, exe: Path) -> None:
        """Every grant a command needs (called under :func:`acl_lock`)."""
        self._grant(name, sid, ws.root, "modify")
        self._grant(name, sid, state, "modify")
        for p in policy.extra_paths("write"):
            self._grant(name, sid, p, "modify")
        prot = policy.protected_paths(ws)
        for p in prot.readonly:
            self._grant(name, sid, p, "deny_write")
        for p in prot.hidden:
            self._grant(name, sid, p, "deny_read")
        readable = [*toolchain_dirs(exe), *policy.extra_paths("read")]
        for p in readable:
            self._grant(name, sid, p, "read")
        covered = [Path(os.path.realpath(c)) for c in (ws.root, state, *policy.extra_paths("write"), *readable)]
        for p in parent_dirs([ws.root, *readable]):
            if (name, p) in self._no_traverse or any(p == c or c in p.parents for c in covered):
                continue  # already reachable through a broader grant (one entry per path in the store)
            try:
                self._grant(name, sid, p, "traverse")
            except OSError as e:  # e.g. a drive root the user may not change: tools may still work without it
                self._no_traverse.add((name, p))
                logger.info("No list access on a parent folder", extra={"path": str(p), "error": str(e)})

    def prepare(self, run: SandboxRun) -> Launch:
        args = unwrap_launcher(run.args) if isinstance(run.args, list) else run.args
        ws = Workspace.create(run.workspace)
        name = container_name(ws.root)
        sid = self._sid(name)
        state = policy.state_dir(ws.root)
        exe = Path(args.split('"')[1]) if isinstance(args, str) else Path(args[0])
        try:
            with acl_lock():
                self.grants.reload()
                self._grant_all(name, sid, ws, state, exe)
        except OSError as e:
            raise SandboxError(f"could not set up the sandbox's file access: {e}") from e

        capabilities = [Sid.from_string(_INTERNET_CLIENT)] if run.network == "full" else []
        limits = settings.sandbox.limits
        env = policy.sandbox_env(run.env, state, run.network)
        try:
            shim = str(python_shim(state))
            env["PYTHONPATH"] = os.pathsep.join(p for p in (shim, env.get("PYTHONPATH", "")) if p)
        except OSError as e:  # Python still runs; only 0o700 temp folders stay unusable
            logger.warning("Could not write the sandbox's Python shim", extra={"error": str(e)})

        def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            return AppContainerPopen(*args, container_sid=sid, capabilities=capabilities, **kwargs)

        return Launch(args, run.cwd, env, popen=popen, job_limits=JobLimits(limits.memory_mb, limits.max_processes))

    def _revoke(self, name: str) -> list[str]:
        """Remove the recorded entries and the profile of one container."""
        with acl_lock():
            self.grants.reload()
            return self._revoke_locked(name)

    def _revoke_locked(self, name: str) -> list[str]:
        cleaned: list[str] = []
        entries = self.grants.remove_container(name)
        with self._lock:
            sid = self._sids.pop(name, None) or ensure_profile(name)
        for entry in entries.values():
            path = Path(entry["path"])
            if not path.exists():
                continue
            try:
                if entry["kind"] == "traverse":
                    set_access_here(path, sid, 0, _REVOKE)
                elif entry["kind"] in _MASKS:
                    set_access(path, sid, 0, _REVOKE)
                else:
                    unprotect(path, sid)
                cleaned.append(str(path))
            except OSError as e:
                logger.warning("Could not remove sandbox access", extra={"path": str(path), "error": str(e)})
        delete_profile(name)
        return cleaned

    def forget(self, root: Path) -> list[str]:
        """Undo the sandbox's access for one workspace."""
        return self._revoke(container_name(Path(os.path.realpath(root))))

    def reset(self) -> list[str]:
        """Remove every recorded entry and AppContainer profile. Returns the paths cleaned."""
        cleaned = [p for name in list(self.grants.data) for p in self._revoke(name)]
        delete_profile("kartrix.probe")
        return cleaned


__all__ = ["AppContainerBackend", "SandboxError"]
