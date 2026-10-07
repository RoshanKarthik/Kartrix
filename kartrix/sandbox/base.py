"""What every sandbox backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kartrix.tools.process_runner import Launch

# off: no network (loopback only where the backend allows it); registries: only the allow-listed
# package registries, through Kartrix's proxy; full: any host (approved network commands, and
# installs on backends that can't limit the network to the registries).
Network = Literal["off", "registries", "full"]


class SandboxUnavailable(Exception):
    """The backend can't run commands here (missing binary, kernel feature, OS version …)."""


class SandboxError(Exception):
    """A sandboxed command couldn't be prepared. Safe to show to the model."""


@dataclass(frozen=True)
class SandboxRun:
    """One command to run in the sandbox (already allowed by the command policy)."""

    args: list[str] | str  # what would run unsandboxed (Decision.run_args)
    argv: list[str]  # the parsed command, program as the model wrote it
    cwd: Path
    env: dict[str, str]  # secrets already removed
    network: Network
    workspace: Path


class Backend(ABC):
    name: str = ""
    registries_enforced: bool = False  # True: network "registries" really limits hosts (proxy)
    loopback: bool = True  # local servers reachable from the command while network is off

    @abstractmethod
    def prepare(self, run: SandboxRun) -> Launch:
        """How to start ``run`` inside the sandbox. Raises :class:`SandboxError`."""

    def describe(self) -> str:
        """One line for the CLI, e.g. "AppContainer — network off; installs need approval"."""
        net = "only the allow-listed registries for installs" if self.registries_enforced else "installs need approval"
        return f"{self.name} — writes limited to the workspace, network off ({net})"

    def label(self, network: Network) -> str:
        """Short note for approvals and the audit log, e.g. "sandboxed: Seatbelt, network off"."""
        return f"sandboxed: {self.name}, network {network}"
