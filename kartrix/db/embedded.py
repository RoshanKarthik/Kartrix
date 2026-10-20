"""Kartrix's own Postgres: used when ``DATABASE_URL`` is not set, so nobody has to install a database or Docker.

The ``embedded-postgres`` wheel ships PostgreSQL 18 with pgvector. Kartrix keeps one server per user in
``<user data dir>/postgres`` and shares it between its processes: the library counts the processes using it and
stops the server when the last one exits. On first use Kartrix

- sets a random password for the ``postgres`` role (kept in ``<user data dir>/postgres/password``, owner-only) and
  requires it for every connection (``scram-sha-256``) — on Windows the server listens on a 127.0.0.1 port, and
  without a password any program on the machine could connect;
- creates the ``kartrix`` database; the migrations then add pgvector and the tables.

``DATABASE_URL`` is then set for this process, so child processes (``kartrix serve`` workers) connect to the same
server instead of starting their own. Sandboxed commands never see it (it is a secret: kartrix.security.environment).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from kartrix.observability.logger import get_logger
from kartrix.paths import user_data_dir

logger = get_logger(__name__)

DATABASE = "kartrix"
ROLE = "postgres"
_HBA = (
    "# Written by Kartrix: every connection needs the password (see kartrix/db/embedded.py).\n"
    "local   all   all                  scram-sha-256\n"
    "host    all   all   127.0.0.1/32   scram-sha-256\n"
    "host    all   all   ::1/128        scram-sha-256\n"
)

_lock = threading.Lock()
_server: Any = None  # this process's handle; released (and the server stopped if unused) at exit


class EmbeddedDatabaseError(RuntimeError):
    """The built-in database could not be started or set up."""


def root() -> Path:
    path = user_data_dir() / "postgres"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_running_here() -> bool:
    """True when this process uses the built-in database (it started or joined it)."""
    return _server is not None


def ensure() -> str:
    """Start (or join) the built-in database, set it up on first use, bring the schema up to date, and return its
    URL (also exported as ``DATABASE_URL``). Takes a few seconds the first time, about a second after that."""
    global _server
    with _lock:
        if _server is not None:
            return os.environ["DATABASE_URL"]
        try:
            import embedded_postgres  # noqa: F401 — installed? (used by the helpers below)
            import fasteners
        except ImportError as e:  # a trimmed install
            raise EmbeddedDatabaseError(f"the built-in database isn't installed ({e}); set DATABASE_URL") from e
        base = root()
        try:
            with fasteners.InterProcessLock(str(base / ".setup.lock")):
                _forget_dead_users(base / "data")
                server = _start_or_join(base / "data")
                password = _set_up(server, base)
                _ensure_answering(server, password)
        except EmbeddedDatabaseError:
            raise
        except Exception as e:
            raise EmbeddedDatabaseError(
                f"the built-in database didn't start: {e} (log: {base / 'data' / 'log'})"
            ) from e
        url = _url(server.get_postmaster_info(), password)
        os.environ["DATABASE_URL"] = url
        _server = server
        logger.info("Built-in database ready", extra={"pgdata": str(base / "data")})
    from kartrix.db.migrate import upgrade_to_head

    upgrade_to_head()
    return url


def _set_up(server: Any, base: Path) -> str:
    """First use: password, password-only access, the ``kartrix`` database. Safe to repeat after a crash."""
    password_file = base / "password"
    hba = base / "data" / "pg_hba.conf"
    if password_file.exists() and hba.read_text(encoding="utf-8") == _HBA:
        return password_file.read_text(encoding="utf-8").strip()
    if password_file.exists():
        password = password_file.read_text(encoding="utf-8").strip()
    else:
        password = secrets.token_urlsafe(24)  # [A-Za-z0-9_-]: safe in a URL and an SQL literal
        _write_private(password_file, password)
    # Still on the trust rules initdb wrote: set the password, create the database, then lock it down. asyncpg, not
    # the library's psql helper: that one pipes psql's output through a shell and hangs on Windows.
    info = server.get_postmaster_info()

    async def run() -> None:
        import asyncpg  # type: ignore[import-untyped]

        conn = await asyncpg.connect(**_connect_args(info), user=ROLE, database="postgres", timeout=30)
        try:
            await conn.execute(f"ALTER ROLE {ROLE} PASSWORD '{password}'")
            if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", DATABASE):
                await conn.execute(f"CREATE DATABASE {DATABASE}")
            hba.write_text(_HBA, encoding="utf-8")
            await conn.execute("SELECT pg_reload_conf()")
        finally:
            await conn.close()

    _run_async(run())
    return password


RECOVERY_SECONDS = 300.0


def _start_or_join(pgdata: Path, wait_seconds: float = RECOVERY_SECONDS) -> Any:
    """The library's handle on the server, started if needed. ``pg_ctl start`` gets only 10 s from the library, and
    after an unclean shutdown (a crash, a power cut) PostgreSQL first recovers — that took 36 s here — so on a
    timeout wait for the server to say it is ready, then join it."""
    import subprocess

    import embedded_postgres
    from embedded_postgres.postgres_server import PostgresServer
    from embedded_postgres.utils import PostmasterInfo

    try:
        return embedded_postgres.get_server(pgdata, cleanup_mode="stop")
    except subprocess.TimeoutExpired:
        logger.info("Built-in database is recovering from an unclean shutdown; waiting for it")
        resolved = pgdata.expanduser().resolve()
        PostgresServer._instances.pop(resolved, None)  # the library keeps the half-made handle; start over
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            info = PostmasterInfo.read_from_pgdata(resolved)
            if info is not None and info.is_running() and info.status == "ready":
                return embedded_postgres.get_server(pgdata, cleanup_mode="stop")
            time.sleep(1)
        raise


def _forget_dead_users(pgdata: Path) -> None:
    """Drop processes that no longer exist from the library's list of the server's users. A Kartrix process that
    was killed never removes itself, and the server is only stopped when that list empties: it would run forever."""
    from embedded_postgres.utils import process_is_running

    users = pgdata / ".handle_pids.json"
    try:
        pids = json.loads(users.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(pids, list):
        return
    alive = [pid for pid in pids if isinstance(pid, int) and process_is_running(pid)]
    if alive != pids:
        users.write_text(json.dumps(alive), encoding="utf-8")
        logger.info("Forgot built-in database users that no longer run", extra={"gone": len(pids) - len(alive)})


def _answers(server: Any, password: str) -> bool:
    """True when the server accepts a login and a query within a few seconds."""

    async def probe() -> None:
        import asyncpg

        conn = await asyncpg.connect(
            **_connect_args(server.get_postmaster_info()), user=ROLE, password=password, database=DATABASE, timeout=10
        )
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()

    try:
        _run_async(probe())
    except Exception as e:
        logger.warning("Built-in database doesn't answer", extra={"error": f"{type(e).__name__}: {e}"})
        return False
    return True


def _restart(server: Any) -> None:
    """Stop the server at once (it doesn't serve anyone) and start it again."""
    import embedded_postgres

    pg_ctl = vars(embedded_postgres)["pg_ctl"]  # generated at import time by the library
    try:
        pg_ctl(["-w", "stop", "-m", "immediate"], pgdata=server.pgdata, timeout=60)
    except Exception as e:  # pg_ctl couldn't: end the postmaster itself
        logger.warning("pg_ctl stop failed; ending the server process", extra={"error": repr(e)})
        process = server.get_postmaster_info().process
        process.kill()
        process.wait(10)
    with server._lock:
        server.ensure_postgres_running()


def _ensure_answering(server: Any, password: str, answers: Any = _answers, restart: Any = _restart) -> None:
    """A server joined after a crash can be running but unable to serve: on Windows a crash-restart can leave
    PostgreSQL unable to start new backends ("could not reserve shared memory region ... error code 487"), so every
    connection is dropped. Restart it once; still nothing means the user has to look at the log."""
    if answers(server, password):
        return
    logger.warning("Restarting the built-in database")
    restart(server)
    if not answers(server, password):
        raise EmbeddedDatabaseError(f"the built-in database doesn't accept connections (log: {server.pgdata / 'log'})")


def _connect_args(info: Any) -> dict[str, Any]:
    if getattr(info, "socket_dir", None):
        return {"host": str(info.socket_dir), "port": info.port or 5432}
    return {"host": "127.0.0.1", "port": info.port}


def _run_async(coro: Any) -> None:
    """Run ``coro`` to completion in a thread of its own (the caller may already be inside an event loop)."""
    import asyncio

    errors: list[BaseException] = []

    def target() -> None:
        try:
            asyncio.run(coro)
        except BaseException as e:
            errors.append(e)

    worker = threading.Thread(target=target, name="kartrix-db-setup")
    worker.start()
    worker.join()
    if errors:
        raise errors[0]


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def _url(info: Any, password: str) -> str:
    from sqlalchemy.engine import URL

    if getattr(info, "socket_dir", None):  # macOS / Linux: a Unix socket, no TCP port open at all
        query = {"host": str(info.socket_dir), "port": str(info.port or 5432)}
        url = URL.create("postgresql+asyncpg", username=ROLE, password=password, database=DATABASE, query=query)
    else:  # Windows: 127.0.0.1 on a free port
        url = URL.create(
            "postgresql+asyncpg", username=ROLE, password=password, host="127.0.0.1", port=info.port, database=DATABASE
        )
    return url.render_as_string(hide_password=False)
