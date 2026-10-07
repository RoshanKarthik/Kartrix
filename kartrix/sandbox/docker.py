"""Optional Docker backend (``sandbox.backend: docker``) — never required.

The command runs in a throw-away container of ``sandbox.docker.image``: the workspace mounted at
``/workspace`` (protected paths re-mounted read-only or hidden), no capabilities, no new
privileges, memory/process/CPU limits, and ``--network none`` unless the run needs the network
(then the default bridge — Docker can't limit it to the registries, so installs need approval).
The host's toolchain isn't visible: the image must contain it, and the program is looked up on the
image's PATH. Paths outside the workspace in arguments don't exist inside.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.sandbox import policy
from kartrix.sandbox.base import Backend, SandboxError, SandboxRun, SandboxUnavailable
from kartrix.security.command_rules import program_name
from kartrix.security.environment import scrubbed_env
from kartrix.security.workspace import Workspace
from kartrix.tools.process_runner import Launch

logger = get_logger(__name__)

_MOUNT = PurePosixPath("/workspace")


def _container_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return str(_MOUNT / rel) if rel != "." else str(_MOUNT)


class DockerBackend(Backend):
    name = "Docker"
    registries_enforced = False

    def __init__(self, docker: str, image: str) -> None:
        self.docker = docker
        self.image = image

    @classmethod
    def detect(cls) -> DockerBackend:
        image = settings.sandbox.docker.image
        if not image:
            raise SandboxUnavailable("sandbox.docker.image is not set (an image with your project's toolchain)")
        docker = shutil.which("docker")
        if docker is None:
            raise SandboxUnavailable("docker is not installed")
        try:
            proc = subprocess.run(  # noqa: S603 — docker from PATH, fixed arguments
                [docker, "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=15, check=False, env=scrubbed_env(),
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SandboxUnavailable(f"docker doesn't answer ({e})") from e
        if proc.returncode != 0:
            raise SandboxUnavailable("the Docker daemon isn't running")
        return cls(docker, image)

    def describe(self) -> str:
        return f"Docker ({self.image}) — only the workspace is mounted, network off; installs need approval"

    def prepare(self, run: SandboxRun) -> Launch:
        ws = Workspace.create(run.workspace)
        state = policy.state_dir(ws.root)
        prot = policy.protected_paths(ws)
        empty = state / "home" / "empty"
        empty.write_text("", encoding="utf-8")
        limits = settings.sandbox.limits
        name = f"kartrix-{uuid.uuid4().hex[:12]}"
        args = [self.docker, "run", "--rm", "--init", "--name", name,
                "--network", "bridge" if run.network != "off" else "none",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--tmpfs", "/tmp:exec", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp",  # noqa: S108 — inside the container
                "-v", f"{ws.root}:{_MOUNT}"]  # fmt: skip
        for p in prot.readonly:
            args += ["-v", f"{p}:{_container_path(ws.root, p)}:ro"]
        for p in prot.hidden:
            target = _container_path(ws.root, p)
            args += ["--tmpfs", f"{target}:ro"] if p.is_dir() else ["-v", f"{empty}:{target}:ro"]
        if limits.memory_mb:
            args += ["--memory", f"{limits.memory_mb}m"]
        if limits.max_processes:
            args += ["--pids-limit", str(limits.max_processes)]
        if settings.sandbox.docker.cpus:
            args += ["--cpus", str(settings.sandbox.docker.cpus)]
        getuid, getgid = getattr(os, "getuid", None), getattr(os, "getgid", None)
        if getuid is not None and getgid is not None:  # POSIX hosts: files keep the user's owner
            args += ["--user", f"{getuid()}:{getgid()}"]
        args += ["-w", _container_path(ws.root, run.cwd), self.image, *self._argv(run, ws)]

        def remove() -> None:
            subprocess.run(  # noqa: S603 — docker from PATH, generated container name
                [self.docker, "rm", "-f", name], capture_output=True, timeout=30, check=False, env=scrubbed_env()
            )

        return Launch(args, run.cwd, scrubbed_env(), on_kill=remove)

    @staticmethod
    def _argv(run: SandboxRun, ws: Workspace) -> list[str]:
        first = run.argv[0]
        if "/" in first or "\\" in first:  # a program inside the workspace: same file, container path
            real = Path(os.path.realpath(run.cwd / first))
            try:
                first = _container_path(ws.root, real)
            except ValueError:
                raise SandboxError(
                    f"{run.argv[0]} is outside the workspace and doesn't exist in the container"
                ) from None
        else:
            first = program_name(first)
        return [first, *run.argv[1:]]
