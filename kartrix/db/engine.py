"""Async SQLAlchemy engine and session factory for Postgres.

The connection URL (it contains the password) is read from the ``DATABASE_URL``
env var; pool tuning comes from the ``database`` section of config.yaml.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from kartrix.config import settings
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class DatabaseConfigError(RuntimeError):
    """Raised when DATABASE_URL is missing or not an async Postgres URL."""


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseConfigError("DATABASE_URL is not set — add it to .env (see .env.example)")
    if make_url(url).drivername != "postgresql+asyncpg":
        raise DatabaseConfigError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
    return url


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        url = get_database_url()
        cfg = settings.database
        _engine = create_async_engine(
            url,
            pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow,
            pool_timeout=cfg.pool_timeout,
            pool_pre_ping=True,
            echo=cfg.echo,
        )
        # render_as_string hides the password in the log line.
        logger.info("Database engine created", extra={"url": make_url(url).render_as_string(hide_password=True)})
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Unit of work: commit on success, roll back on any exception."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close all pooled connections (call on shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
