"""MCP server definitions (``kartrix/mcp_servers.json``), validated strictly (B7).

Every server must be **pinned**:

- ``binary`` — a release archive per platform with its SHA-256 (``kartrix.mcp.binaries``), or
- ``command`` — ``npx <pkg>@<exact version>`` / ``uvx <pkg>==<exact version>``.

Anything else is refused unless the entry says ``"allow_unpinned": true`` (logged at startup).
``${VAR}`` placeholders are only substituted in ``env`` values; a variable that is unset leaves
the entry out (e.g. no ``GITHUB_PERSONAL_ACCESS_TOKEN`` → the GitHub server logs in via the browser).
Unknown keys are errors, so a typo can't silently disable a restriction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kartrix.mcp.binaries import Asset, BinarySpec

_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"

_SEMVER = r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?"
_NPM_PINNED = re.compile(rf"^(?:@[\w.-]+/)?[\w.-]+@{_SEMVER}$")
_PY_PINNED = re.compile(rf"^[A-Za-z0-9][\w.-]*(?:\[[\w,.-]+\])?(?:==|@){_SEMVER}$")
_VAR = re.compile(r"\$\{(\w+)\}")


class McpConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetModel(_Strict):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class BinaryModel(_Strict):
    name: str = Field(pattern=r"^[\w.-]+$")
    version: str = Field(pattern=rf"^{_SEMVER}$")
    url: str
    executable: str = Field(pattern=r"^[\w.-]+$")
    assets: dict[str, AssetModel] = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("binary downloads must use https://")
        return value

    def spec(self) -> BinarySpec:
        assets = {k: Asset(a.file, a.sha256) for k, a in self.assets.items()}
        return BinarySpec(self.name, self.version, self.url, self.executable, assets)


class ToolsModel(_Strict):
    allow: Literal["*"] | list[str] = "*"  # tools the agent may see
    approve: list[str] = []  # tools that always need the user's approval


class McpServer(_Strict):
    description: str = ""
    binary: BinaryModel | None = None
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    tools: ToolsModel = ToolsModel()
    allow_unpinned: bool = False

    @model_validator(mode="after")
    def _pinned(self) -> McpServer:
        if (self.binary is None) == (self.command is None):
            raise ValueError("set exactly one of 'binary' or 'command'")
        if self.command is not None and not self.allow_unpinned and not command_is_pinned(self.command, self.args):
            raise ValueError(
                f"'{self.command}' server is not pinned to an exact version (npx pkg@1.2.3 / uvx pkg==1.2.3); "
                "pin it, or set allow_unpinned: true if you accept the risk"
            )
        return self

    def resolved_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.env.items():
            resolved = _VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)
            if resolved:
                out[key] = resolved
        return out


def _positional(args: list[str], value_flags: set[str]) -> tuple[list[str], list[str]]:
    """(positional args, values of value_flags) — flags that take a value consume the next arg."""
    positional: list[str] = []
    values: list[str] = []
    it = iter(args)
    for arg in it:
        if arg in value_flags:
            values.append(next(it, ""))
        elif arg.startswith("-"):
            name, _, inline = arg.partition("=")
            if name in value_flags and inline:
                values.append(inline)
        else:
            positional.append(arg)
    return positional, values


def command_is_pinned(command: str, args: list[str]) -> bool:
    launcher = Path(command).name.lower().removesuffix(".cmd").removesuffix(".exe")
    if launcher == "npx":
        positional, packages = _positional(args, {"-p", "--package"})
        specs = packages or positional[:1]
        return bool(specs) and all(_NPM_PINNED.match(s) for s in specs)
    if launcher == "uvx":
        positional, sources = _positional(args, {"--from", "--with", "--python", "-p"})
        specs = [v for v in sources if not re.match(r"^\d", v)] or positional[:1]  # --python 3.12 isn't a package
        return bool(specs) and all(_PY_PINNED.match(s) for s in specs)
    return False


def load_mcp_configs(path: Path | None = None) -> dict[str, McpServer]:
    """All configured servers, validated. Raises McpConfigError with every problem found."""
    raw: Any = json.loads((path or _CONFIG_PATH).read_text(encoding="utf-8"))
    servers = raw.get("mcp_servers", {}) if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        raise McpConfigError("mcp_servers.json must contain an object 'mcp_servers'")
    out: dict[str, McpServer] = {}
    errors: list[str] = []
    for name, entry in servers.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name):
            errors.append(f"{name!r}: server names are lowercase letters, digits, '-' and '_'")
            continue
        try:
            out[name] = McpServer.model_validate(entry)
        except ValidationError as e:
            errors.extend(f"{name}: {err['loc']}: {err['msg']}" for err in e.errors())
    if errors:
        raise McpConfigError("invalid MCP server config:\n  " + "\n  ".join(errors))
    return out
