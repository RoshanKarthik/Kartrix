"""Run a vetted command and make sure nothing it started outlives it.

- Windows: the process is placed in a Job Object with KILL_ON_JOB_CLOSE. Children join the
  job automatically, so terminating/closing the job kills the whole tree — even processes
  the command left running in the background.
- POSIX: the process gets its own session (process group); the group is killed afterwards.

stdin is closed, so a program waiting for input fails instead of hanging. With ``should_stop``
(the kill switch, :mod:`kartrix.security.budget`) the command is polled every half second and
its whole tree is killed as soon as the run must stop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stopped: str | None = None  # why the kill switch killed it


_POLL_SECONDS = 0.5

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", ctypes.c_uint64 * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_EXTENDED_LIMIT_INFORMATION = 9
    _KILL_ON_JOB_CLOSE = 0x2000

    def _create_job(proc: subprocess.Popen[bytes]) -> int | None:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            job, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ) and _kernel32.AssignProcessToJobObject(job, int(proc._handle))  # type: ignore[attr-defined]
        if not ok:
            logger.warning("Could not put command in a job object", extra={"error": ctypes.get_last_error()})
            _kernel32.CloseHandle(job)
            return None
        return int(job)

    def _end_tree(proc: subprocess.Popen[bytes], job: int | None) -> None:
        if job is not None:
            _kernel32.TerminateJobObject(job, 1)
            _kernel32.CloseHandle(job)
            return
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        subprocess.run(  # noqa: S603 — fixed system binary, numeric pid
            [str(taskkill), "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10, check=False
        )

    _POPEN_KW: dict[str, object] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

else:

    def _create_job(proc: subprocess.Popen[bytes]) -> int | None:
        return proc.pid  # the process leads its own group (start_new_session)

    def _end_tree(proc: subprocess.Popen[bytes], job: int | None) -> None:
        try:
            os.killpg(job or proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    _POPEN_KW = {"start_new_session": True}


def run_process(
    args: list[str] | str,
    cwd: Path | None,
    env: dict[str, str],
    timeout: float,
    should_stop: Callable[[], str | None] | None = None,
) -> ProcessResult:
    """Run ``args`` (``shell=False``; a str only for pre-quoted Windows batch command lines).
    ``should_stop`` returns a reason once the command must be killed (polled every half second)."""
    proc = subprocess.Popen(  # noqa: S603 — args produced and vetted by the command policy
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        **_POPEN_KW,  # type: ignore[call-overload]
    )
    job = _create_job(proc)
    timed_out = False
    stopped: str | None = None
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                out, err = proc.communicate(timeout=min(remaining, _POLL_SECONDS) if should_stop else remaining)
                break
            except subprocess.TimeoutExpired:
                stopped = should_stop() if should_stop else None
                timed_out = not stopped and time.monotonic() >= deadline
                if stopped or timed_out:
                    _end_tree(proc, job)
                    job = None
                    proc.kill()
                    out, err = proc.communicate()
                    break
    finally:
        if proc.poll() is None:  # interrupted (e.g. Ctrl+C): don't leave it running
            _end_tree(proc, job)
            job = None
            proc.kill()
    if job is not None:
        _end_tree(proc, job)  # background leftovers die with the command
    return ProcessResult(proc.returncode, _decode(out), _decode(err), timed_out, stopped)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")
