"""SQLAlchemy 2.0 async engine + session factory for Postgres.

Sits alongside ``core.database`` (aiosqlite) during the Phase 2 migration.
Once tenant-scoping and auth are in place, the SQLite layer is retired.

Design notes:
- ``DB_BACKEND=postgres`` activates this module's engine; otherwise the
  legacy aiosqlite layer in ``core.database`` is used. This lets the
  migration land incrementally without breaking the dev SQLite path.
- ``DATABASE_URL`` must be a SQLAlchemy async URL
  (``postgresql+asyncpg://user:pw@host:port/db``).
- The session factory is `async_sessionmaker` so callers do
  ``async with get_session() as s: ...``.
- Models live in ``core.models``; this module only owns the connection.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import settings

log = logging.getLogger("gutenberg.db.sa")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Lazy-create the async engine on first call."""
    global _engine
    if _engine is None:
        url = settings.database_url
        if not url:
            raise RuntimeError(
                "DATABASE_URL is empty — cannot init Postgres engine. "
                "Set DATABASE_URL or run with DB_BACKEND=sqlite."
            )
        _engine = create_async_engine(
            url,
            echo=settings.db_echo,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        log.info(f"async engine created (echo={settings.db_echo})")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an AsyncSession scoped to the request."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    """Tear down the engine on app shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
