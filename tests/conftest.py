"""Shared fixtures.

Database tests use a separate database (``<DATABASE_URL db>_test``, or ``TEST_DATABASE_URL``)
that is created and migrated with Alembic once per run, so dev data is never touched.
Tests needing Postgres/Redis are skipped when the service is unreachable, unless
``KARTRIX_REQUIRE_SERVICES=1`` (set in CI) turns the skip into a failure.
No test calls an LLM or embedding API: ``fake_embedder`` replaces the real one.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent

# Resolve the test database URL *before* kartrix is imported (kartrix.config loads .env,
# but never overrides variables that are already set).
_env = {**dotenv_values(ROOT / ".env"), **os.environ}
_base_url = _env.get("DATABASE_URL")
TEST_DATABASE_URL = _env.get("TEST_DATABASE_URL") or (
    make_url(_base_url).set(database=f"{make_url(_base_url).database}_test").render_as_string(hide_password=False)
    if _base_url
    else None
)
if TEST_DATABASE_URL:
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from tests.fakes import HashingEmbeddings  # noqa: E402 — after DATABASE_URL is set
from tests.helpers import unavailable  # noqa: E402

# ── Postgres ──────────────────────────────────────────────────────────

_TABLES = "code_chunks, code_files, checkpoint_writes, checkpoints, tasks, approvals, projects, sessions"


async def _create_database(url: str) -> None:
    import asyncpg

    target = make_url(url)
    admin = target.set(drivername="postgresql", database="postgres").render_as_string(hide_password=False)
    conn = await asyncpg.connect(admin, timeout=5)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target.database):
            await conn.execute(f'CREATE DATABASE "{target.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def migrated_db() -> str:
    """Create the test database if needed and bring it to the latest migration."""
    if not TEST_DATABASE_URL:
        unavailable("DATABASE_URL is not set")
    assert TEST_DATABASE_URL
    try:
        asyncio.run(_create_database(TEST_DATABASE_URL))
    except Exception as e:
        unavailable(f"Postgres unavailable: {e}")
    env = {**os.environ, "DATABASE_URL": TEST_DATABASE_URL}
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return TEST_DATABASE_URL


@pytest.fixture
async def db(migrated_db: str) -> AsyncIterator[None]:
    """Clean tables before each test; dispose the engine after (it is bound to the test's loop)."""
    from sqlalchemy import text

    from kartrix.db.engine import dispose_engine, session_scope

    async with session_scope() as s:
        # audit_log is append-only by design (TRUNCATE is blocked), so it is left alone.
        await s.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
    yield
    await dispose_engine()


# ── Embeddings / Redis ────────────────────────────────────────────────


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> HashingEmbeddings:
    """Replace the NIM embedder everywhere it is looked up."""
    emb = HashingEmbeddings()
    for mod in (
        "kartrix.context.indexers.pg_index",
        "kartrix.context.retrievers.pg_hybrid",
        "kartrix.cache.semantic_cache",
    ):
        monkeypatch.setattr(f"{mod}.get_embedder", lambda: emb)
    return emb


@pytest.fixture
def redis_url() -> str:
    url = _env.get("REDIS_URL")
    if not url:
        unavailable("REDIS_URL is not set")
    assert url
    return url


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Path]:
    """An empty directory to build a fake repository in."""
    yield tmp_path / "repo"
