"""Kartrix's built-in database (kartrix.db.embedded): a real PostgreSQL + pgvector from the ``embedded-postgres``
wheel, started in a temporary user-data folder — set up with a password, migrated, and refusing connections
without the password. Slow (tens of seconds): it creates a database cluster."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kartrix.db import embedded


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("KARTRIX_HOME", str(home))
    monkeypatch.setenv("DATABASE_URL", "")  # empty, not unset: the repo's .env must not supply one
    monkeypatch.setattr(embedded, "_server", None)
    yield home
    server = embedded._server
    if server is not None:
        server.cleanup()  # stops the server (this process was its only user)


async def _connect(url: str, password: str | None) -> Any:
    import asyncpg  # type: ignore[import-untyped]
    from sqlalchemy.engine import make_url

    u = make_url(url)
    host = u.query.get("host") or u.host
    port = int(u.query.get("port") or u.port or 5432)
    return await asyncpg.connect(host=host, port=port, user=u.username, password=password, database=u.database)


def test_built_in_database_end_to_end(isolated: Path) -> None:
    import os

    url = embedded.ensure()
    assert os.environ["DATABASE_URL"] == url and embedded.is_running_here()
    password = (embedded.root() / "password").read_text(encoding="utf-8").strip()
    assert embedded.root().is_relative_to(isolated)  # under the user's data folder, nowhere else

    async def check() -> None:
        conn = await _connect(url, password)
        try:
            assert await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            assert await conn.fetchval("SELECT count(*) FROM alembic_version") == 1  # migrated
            assert await conn.fetchval("SELECT to_regclass('public.code_chunks') IS NOT NULL")
        finally:
            await conn.close()
        with pytest.raises(Exception, match="password"):  # no password, no access
            await _connect(url, None)
        with pytest.raises(Exception, match="password"):
            await _connect(url, "wrong-password")

    asyncio.run(check())
    assert embedded.ensure() == url  # a second call reuses this process's server


def test_database_url_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.db.engine import get_database_url

    monkeypatch.setattr(embedded, "_server", None)
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@db.example:5432/k")
    assert get_database_url() == "postgresql+asyncpg://u:p@db.example:5432/k"
    assert not embedded.is_running_here()  # never started when a database is configured
