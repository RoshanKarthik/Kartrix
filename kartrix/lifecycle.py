"""``kartrix update`` and ``kartrix uninstall`` for installs made with ``uv tool install`` (the install scripts).

On Windows the running ``kartrix.exe`` and its Python files are locked while Kartrix runs, so uv can't replace or
remove them from inside this process: the uv command is started in a new console window that waits until Kartrix
has exited. On macOS/Linux files in use can be replaced, so uv simply takes over this process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def uv_executable() -> str | None:
    return shutil.which("uv")


def tool_dir(uv: str) -> Path | None:
    """Where ``uv tool`` keeps its tools (``uv tool dir``), or None if uv can't say."""
    try:
        out = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, timeout=30, check=True)  # noqa: S603
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.stdout.strip()) if out.stdout.strip() else None


def installed_as_tool(tools: Path | None, prefix: Path | None = None) -> bool:
    """True when this Python environment is the ``kartrix`` tool uv installed (not a development checkout)."""
    if tools is None:
        return False
    prefix = (prefix or Path(sys.prefix)).resolve()
    try:
        return prefix.parent == tools.resolve() and prefix.name.lower() == "kartrix"
    except OSError:
        return False


def handoff_command(uv: str, args: list[str], platform: str | None = None) -> list[str]:
    """The command that runs ``uv <args>`` once this process is gone."""
    if (platform or sys.platform) == "win32":
        inner = f'timeout /t 3 /nobreak >nul & "{uv}" {" ".join(args)} & echo. & pause'
        return ["cmd.exe", "/c", inner]
    return [uv, *args]


def _hand_off(uv: str, args: list[str]) -> int:
    platform = sys.platform  # a variable: type-checks every branch on every OS
    command = handoff_command(uv, args, platform)
    if platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(command, creationflags=flags, close_fds=True)  # noqa: S603 — uv, fixed arguments
        print("A new window finishes the job in a few seconds.")
        return 0
    os.execv(command[0], command)  # noqa: S606 — replaces this process with uv


def _not_a_tool_install(uv: str | None) -> int:
    if uv is None:
        print("uv isn't on PATH. Reinstall with the install script (it sets up uv), or use pip/pipx as you installed.")
    else:
        print("This Kartrix isn't a `uv tool` install (a development checkout?). Update it the way it was installed,")
        print("e.g. `git pull && uv sync` in the checkout.")
    return 1


def update() -> int:
    uv = uv_executable()
    if uv is None or not installed_as_tool(tool_dir(uv)):
        return _not_a_tool_install(uv)
    print("Updating Kartrix...")
    return _hand_off(uv, ["tool", "upgrade", "kartrix"])


def data_folders() -> list[Path]:
    from kartrix.paths import user_cache_dir, user_config_path, user_data_dir

    return [p for p in dict.fromkeys([user_data_dir(), user_config_path(), user_cache_dir()]) if p.exists()]


def stop_built_in_database() -> None:
    """Stop the built-in database server if one is running from the user's data folder (before deleting it)."""
    from kartrix.paths import user_data_dir

    pgdata = user_data_dir() / "postgres" / "data"
    if not (pgdata / "postmaster.pid").exists():
        return
    try:
        import embedded_postgres

        pg_ctl = vars(embedded_postgres)["pg_ctl"]  # generated at import time by the library
        pg_ctl(["stop", "-m", "fast"], pgdata=pgdata, timeout=60)
    except Exception as e:
        print(f"Couldn't stop the built-in database ({e}); close other Kartrix windows and try again.")


def uninstall(ask: Callable[[str], str] = input) -> int:
    uv = uv_executable()
    folders = data_folders()
    if folders:
        print("Kartrix keeps your data (built-in database, keys, settings, caches) in:")
        for folder in folders:
            print(f"  {folder}")
        answer = ask("Delete this data too? It can't be undone. [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            stop_built_in_database()
            for folder in folders:
                shutil.rmtree(folder, ignore_errors=True)
            print("Your Kartrix data was deleted.")
        else:
            print("Your data stays; reinstalling Kartrix picks it up again.")
    if uv is None or not installed_as_tool(tool_dir(uv)):
        return _not_a_tool_install(uv) if uv is None else _dev_uninstall_hint()
    print("Removing Kartrix...")
    return _hand_off(uv, ["tool", "uninstall", "kartrix"])


def _dev_uninstall_hint() -> int:
    print("This Kartrix isn't a `uv tool` install; remove it the way it was installed (e.g. delete the checkout).")
    return 1
