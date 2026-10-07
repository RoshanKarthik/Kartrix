"""What a sandboxed command may touch — shared by every backend.

- **Writable:** the workspace, except ``workspace.deny_write`` (``.git/``, ``.kartrix/``,
  secret files), plus a per-workspace state folder (temp files, package caches) and
  ``sandbox.extra_write``.
- **Hidden:** the workspace's ``workspace.deny_read`` paths (``.env``, keys …) and
  ``sandbox.deny_read_home`` (``~/.ssh``, cloud credentials, registry tokens, Kartrix's own data).
- **Environment:** temp and cache variables point into the state folder, so ``npm``/``pip``/``uv``
  never write to (or read tokens from) the user's real caches and config files.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from kartrix.config import PROJECT_ROOT, settings
from kartrix.paths import user_cache_dir, user_data_dir
from kartrix.sandbox.base import Network
from kartrix.security.workspace import Workspace

_MAX_PROTECTED = 5000  # protected paths reported per workspace; beyond this the walk stops
_ALWAYS_SKIP = {"node_modules", ".venv", "venv", "__pycache__"}  # big trees nobody keeps secrets in


@dataclass(frozen=True)
class Protected:
    hidden: list[Path]  # can't be read (nor written)
    readonly: list[Path]  # can be read, not written


def workspace_key(root: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:16]


def state_dir(root: Path) -> Path:
    """Per-workspace folder the sandbox may write: ``tmp/``, ``cache/``, ``home/``."""
    path = user_cache_dir() / "sandbox" / workspace_key(root)
    for sub in ("tmp", "cache", "home"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    npmrc = path / "home" / ".npmrc"
    if not npmrc.exists():
        npmrc.write_text("", encoding="utf-8")
    return path


def _skip_dirs() -> set[str]:
    names = {p.strip("/") for p in settings.checkpoints.exclude if p.endswith("/") and "/" not in p.strip("/")}
    return _ALWAYS_SKIP | {n for n in names if n and "*" not in n}


def protected_paths(ws: Workspace) -> Protected:
    """Existing workspace paths matched by ``deny_read`` / ``deny_write``. Directories are
    reported once (not their contents); symlinks are skipped (nothing follows them here)."""
    hidden: list[Path] = []
    readonly: list[Path] = []
    skip = _skip_dirs()

    def walk(directory: Path) -> Iterator[os.DirEntry[str]]:
        try:
            with os.scandir(directory) as it:
                yield from it
        except OSError:
            return

    stack = [ws.root]
    while stack and len(hidden) + len(readonly) < _MAX_PROTECTED:
        for entry in walk(stack.pop()):
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            is_dir = entry.is_dir(follow_symlinks=False)
            if not ws.is_readable(path):
                hidden.append(path)
            elif not ws.is_writable(path):
                readonly.append(path)
                if is_dir:
                    stack.append(path)  # a hidden file can still sit inside a read-only folder
            elif is_dir and entry.name not in skip:
                stack.append(path)
    return Protected(hidden, readonly)


def home_secrets(workspace: Path) -> list[Path]:
    """Existing credential paths outside the workspace that sandboxed commands must not read."""
    home = Path.home()
    found: list[Path] = []
    for rel in settings.sandbox.deny_read_home:
        p = home / rel
        if p.exists():
            found.append(p)
    data = user_data_dir()
    found.append(data)
    for env_file in PROJECT_ROOT.glob(".env*"):  # Kartrix's own keys (development installs)
        if env_file.is_file() and env_file.name not in (".env.example", ".env.sample", ".env.template"):
            found.append(env_file)
    ws = os.path.normcase(str(workspace))
    return [p for p in dict.fromkeys(found) if not os.path.normcase(str(p)).startswith(ws + os.sep)]


def extra_paths(kind: str) -> list[Path]:
    values = settings.sandbox.extra_read if kind == "read" else settings.sandbox.extra_write
    return [Path(os.path.expanduser(v)).resolve() for v in values if v.strip()]


_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
               "npm_config_proxy", "npm_config_https_proxy", "NO_PROXY", "no_proxy")  # fmt: skip


def sandbox_env(env: dict[str, str], state: Path, network: Network, proxy_url: str | None = None) -> dict[str, str]:
    """``env`` with temp/cache folders moved into ``state`` and the proxy set for ``registries``."""
    out = dict(env)
    tmp, cache = str(state / "tmp"), state / "cache"
    out.update(
        {
            "TMPDIR": tmp,
            "TMP": tmp,
            "TEMP": tmp,
            "XDG_CACHE_HOME": str(cache),
            "npm_config_cache": str(cache / "npm"),
            "npm_config_userconfig": str(state / "home" / ".npmrc"),
            "npm_config_store_dir": str(cache / "pnpm-store"),
            "npm_config_update_notifier": "false",
            "YARN_CACHE_FOLDER": str(cache / "yarn"),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "UV_CACHE_DIR": str(cache / "uv"),
        }
    )
    if network != "full":
        for name in _PROXY_VARS:
            out.pop(name, None)
    if network == "registries" and proxy_url:
        for name in _PROXY_VARS[:8]:
            out[name] = proxy_url
        out["NO_PROXY"] = out["no_proxy"] = "localhost,127.0.0.1,::1"
    return out
