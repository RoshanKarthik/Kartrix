"""Run a vetted command and make sure nothing it started outlives it.

- Windows: the process is placed in a Job Object with KILL_ON_JOB_CLOSE. Children join the
  job automatically, so terminating/closing the job kills the whole tree — even processes
  the command left running in the background. Sandboxed commands start suspended and are
  resumed only once they are in the job, which also carries their resource limits.
- POSIX: the process gets its own session (process group); the group is killed afterwards.

stdin is closed, so a program waiting for input fails instead of hanging. With ``should_stop``
(the kill switch, :mod:`kartrix.security.budget`) the command is polled every half second and
its whole tree is killed as soon as the run must stop.

A :class:`Launch` says how to start the command: plain, or wrapped by a sandbox backend
(:mod:`kartrix.sandbox`), which may also supply its own ``Popen`` class and clean-up steps.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stopped: str | None = None  # why the kill switch killed it


@dataclass(frozen=True)
class JobLimits:
    """Limits for the whole process tree (Windows job objects)."""

    memory_mb: int | None = None
    max_processes: int | None = None


@dataclass
class Launch:
    """Everything needed to start one command."""

    args: list[str] | str  # a str only for pre-quoted Windows batch command lines
    cwd: Path | None
    env: dict[str, str]
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen
    popen_kwargs: dict[str, Any] = field(default_factory=dict)
    job_limits: JobLimits | None = None
    on_kill: Callable[[], None] | None = None  # extra clean-up after a kill (e.g. remove a container)
    after: Callable[[ProcessResult], ProcessResult] | None = None  # post-run check of the result


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
    _LIMIT_ACTIVE_PROCESS = 0x8
    _LIMIT_JOB_MEMORY = 0x200
    _DIE_ON_UNHANDLED_EXCEPTION = 0x400  # no crash dialog waiting for a click nobody will make
    _KILL_ON_JOB_CLOSE = 0x2000

    def _create_job(proc: subprocess.Popen[bytes], limits: JobLimits | None = None) -> int | None:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        flags = _KILL_ON_JOB_CLOSE
        if limits is not None:
            flags |= _DIE_ON_UNHANDLED_EXCEPTION
            if limits.memory_mb:
                flags |= _LIMIT_JOB_MEMORY
                info.JobMemoryLimit = limits.memory_mb * 1024 * 1024
            if limits.max_processes:
                flags |= _LIMIT_ACTIVE_PROCESS
                info.BasicLimitInformation.ActiveProcessLimit = limits.max_processes
        info.BasicLimitInformation.LimitFlags = flags
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

    def _create_job(proc: subprocess.Popen[bytes], limits: JobLimits | None = None) -> int | None:
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
    return run_launch(Launch(args, cwd, env), timeout, should_stop)


def run_launch(launch: Launch, timeout: float, should_stop: Callable[[], str | None] | None = None) -> ProcessResult:
    """Start ``launch`` and wait for it, like :func:`run_process`."""
    proc = launch.popen(
        launch.args,
        cwd=launch.cwd,
        env=launch.env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        **{**_POPEN_KW, **launch.popen_kwargs},
    )
    job = _create_job(proc, launch.job_limits)
    resume: Callable[[], None] | None = getattr(proc, "resume", None)  # started suspended (sandbox)
    if resume is not None:
        if job is None:  # its limits and the kill-the-whole-tree guarantee would be missing
            proc.kill()
            proc.communicate()
            raise OSError("could not put the sandboxed command in a job object")
        resume()
    timed_out = False
    stopped: str | None = None
    killed = False
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
                    killed = True
                    out, err = proc.communicate()
                    break
    finally:
        if proc.poll() is None:  # interrupted (e.g. Ctrl+C): don't leave it running
            _end_tree(proc, job)
            job = None
            proc.kill()
            killed = True
        if killed and launch.on_kill is not None:
            launch.on_kill()
    if job is not None:
        _end_tree(proc, job)  # background leftovers die with the command
    result = ProcessResult(proc.returncode, _decode(out), _decode(err), timed_out, stopped)
    return launch.after(result) if launch.after is not None else result


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")
