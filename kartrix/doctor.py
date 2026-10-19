"""``kartrix doctor``: is everything Kartrix needs in place on this machine? Each check says what it found and,
when something is missing, exactly how to fix it. Output is plain ASCII (some Windows consoles can't print
symbols). Exit code 0 when nothing required is missing."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.markup import escape

Status = Literal["ok", "warn", "fail", "off"]
_MARK = {
    "ok": r"[green]\[ ok ][/green]",
    "warn": r"[yellow]\[warn][/yellow]",
    "fail": r"[red]\[FAIL][/red]",
    "off": r"[dim]\[off ][/dim]",
}
_ASCII = str.maketrans({"—": "-", "–": "-", "…": "...", "→": "->", "·": "|"})


def _plain(text: str) -> str:
    """ASCII only: some Windows consoles can't print dashes or arrows."""
    return text.translate(_ASCII).encode("ascii", "replace").decode("ascii")


@dataclass
class Check:
    status: Status
    name: str
    detail: str
    fix: str = ""


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("kartrix")
    except PackageNotFoundError:
        return "unknown"


def check_install() -> Check:
    return Check("ok", "Kartrix", f"{_version()} on Python {sys.version.split()[0]} ({sys.executable})")


def check_keys(online: bool = True) -> list[Check]:
    from kartrix import keys

    saved = keys.read()
    out = []
    for info in keys.KNOWN:
        value = os.environ.get(info.name, "").strip()
        if not value:
            status: Status = "fail" if info.required else "off"
            fix = f"kartrix keys set {info.name}   (free: {info.where})"
            out.append(Check(status, f"{info.provider} key", "not set" + ("" if info.required else " (optional)"), fix))
            continue
        where = "keys file" if saved.get(info.name) == value else "environment / .env"
        if not online:
            out.append(Check("ok", f"{info.provider} key", f"{keys.mask(value)} ({where}, not checked)"))
            continue
        ok, why = keys.validate(info.name, value)
        if ok:
            out.append(
                Check("ok", f"{info.provider} key", f"{keys.mask(value)} ({where}, accepted by {info.provider})")
            )
        elif ok is False:
            fix = f"kartrix keys set {info.name}   (get a new one: {info.where})"
            out.append(Check("fail" if info.required else "warn", f"{info.provider} key", why, fix))
        else:
            out.append(Check("warn", f"{info.provider} key", f"{keys.mask(value)} ({where}; {why})"))
    return out


async def _database_facts() -> tuple[str | None, str | None]:
    """(pgvector version, alembic revision) of the configured database."""
    from sqlalchemy import text

    from kartrix.db.engine import dispose_engine, session_scope

    try:
        async with session_scope() as s:
            vector = (await s.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))).scalar()
            try:
                revision = (await s.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            except Exception:
                revision = None
        return vector, revision
    finally:
        await dispose_engine()


def _folder_size(path: object) -> int:
    from pathlib import Path

    total = 0
    for f in Path(str(path)).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def check_database() -> Check:
    from kartrix.db import embedded
    from kartrix.db.engine import DatabaseConfigError, get_database_url
    from kartrix.db.migrate import MIGRATIONS

    external = bool(os.environ.get("DATABASE_URL", "").strip()) and not embedded.is_running_here()
    try:
        get_database_url()  # starts the built-in database when none is configured
        vector, revision = asyncio.run(_database_facts())
    except (DatabaseConfigError, Exception) as e:
        fix = (
            "check DATABASE_URL"
            if external
            else "kartrix doctor again; if it persists, see the log in the database folder"
        )
        return Check("fail", "Database", f"{type(e).__name__}: {e}"[:300], fix)
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    head = ScriptDirectory.from_config(cfg).get_current_head()
    where = "DATABASE_URL" if external else f"built-in, {embedded.root()} ({_folder_size(embedded.root()) // 2**20} MB)"
    if not vector:
        return Check(
            "fail", "Database", f"{where}: pgvector is not installed", "use a Postgres with the pgvector extension"
        )
    if revision != head:
        return Check(
            "warn",
            "Database",
            f"{where}: schema {revision} (current: {head})",
            "kartrix starts the upgrade itself; for DATABASE_URL run: uv run alembic upgrade head",
        )
    return Check("ok", "Database", f"{where}; pgvector {vector}; schema up to date")


def check_redis() -> Check:
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return Check("off", "Redis", "not set (optional: only the answer cache uses it)")

    async def ping() -> None:
        from redis.asyncio import from_url

        client = from_url(url, socket_connect_timeout=5, socket_timeout=5)
        try:
            await client.ping()  # type: ignore[misc]  # redis types ping() as sync-or-async
        finally:
            await client.aclose()

    try:
        asyncio.run(ping())
    except Exception as e:
        return Check(
            "warn",
            "Redis",
            f"set but unreachable ({type(e).__name__}) — the answer cache stays off",
            "start Redis or remove REDIS_URL",
        )
    return Check("ok", "Redis", "reachable (answer cache on)")


def check_sandbox() -> Check:
    from kartrix.sandbox.manager import status

    current = status()
    if current.backend is not None:
        return Check("ok", "Sandbox", current.describe())
    fix = "on Linux install bubblewrap (e.g. sudo apt install bubblewrap)" if sys.platform.startswith("linux") else ""
    return Check("warn", "Sandbox", current.describe(), fix)


def check_git() -> Check:
    git = shutil.which("git")
    if git:
        return Check("ok", "git", git)
    return Check(
        "warn", "git", "not found — undo checkpoints (/undo) are off", "install git: https://git-scm.com/downloads"
    )


_MOVE = r"free some space, or keep Kartrix's data on a bigger drive: set KARTRIX_HOME (e.g. D:\kartrix-data)"


def check_disk() -> Check:
    from kartrix.paths import user_data_dir

    free = shutil.disk_usage(user_data_dir()).free
    gb = free / 2**30
    where = str(user_data_dir())
    if gb < 0.5:
        return Check("fail", "Disk space", f"{gb:.1f} GB free where Kartrix keeps its data ({where})", _MOVE)
    if gb < 2:
        return Check("warn", "Disk space", f"{gb:.1f} GB free where Kartrix keeps its data ({where})", _MOVE)
    return Check("ok", "Disk space", f"{gb:.1f} GB free ({where})")


def run_doctor(console: Console | None = None, online: bool = True) -> int:
    console = console or Console()
    console.print("[bold]Kartrix doctor[/bold]")
    checks = [
        check_install(),
        *check_keys(online),
        check_database(),
        check_redis(),
        check_sandbox(),
        check_git(),
        check_disk(),
    ]
    for c in checks:
        console.print(f"{_MARK[c.status]} {escape(_plain(c.name))}: {escape(_plain(c.detail))}")
        if c.fix and c.status in ("fail", "warn", "off"):
            console.print(f"       [dim]-> {escape(_plain(c.fix))}[/dim]")
    failed = [c for c in checks if c.status == "fail"]
    console.print("[green]Ready.[/green]" if not failed else f"[red]{len(failed)} problem(s) to fix first.[/red]")
    return 0 if not failed else 1
