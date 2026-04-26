"""One-shot data migration: legacy SQLite → Postgres.

Reads existing /data/gutenberg.db, creates a single ``default@local`` user,
and copies all corpora + documents + conversations + messages under that
user. Idempotent on the user side; raises if rows already exist downstream
to avoid silent double-imports.

Usage (from repo root):

    DATABASE_URL='postgresql+asyncpg://gutenberg:gutenberg_dev@127.0.0.1:5432/gutenberg' \\
        uv run python scripts/sqlite_to_postgres.py /data/gutenberg.db
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import aiosqlite

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "services"))

from sqlalchemy import select  # noqa: E402

from core.db import get_session_factory  # noqa: E402
from core import models  # noqa: E402

DEFAULT_USER_EMAIL = "default@local"
DEFAULT_USER_PASSWORD_HASH = "!disabled"  # placeholder; user must reset


async def get_or_create_default_user(session) -> models.User:
    res = await session.execute(
        select(models.User).where(models.User.email == DEFAULT_USER_EMAIL)
    )
    user = res.scalar_one_or_none()
    if user:
        return user
    user = models.User(
        id=uuid.uuid4(),
        email=DEFAULT_USER_EMAIL,
        hashed_password=DEFAULT_USER_PASSWORD_HASH,
        is_active=False,  # disabled until password is set
        is_superuser=True,
        is_verified=True,
        tier="legacy",
    )
    session.add(user)
    await session.flush()
    print(f"created default user {user.id} ({user.email})")
    return user


async def migrate(sqlite_path: str) -> None:
    if not os.path.isfile(sqlite_path):
        raise SystemExit(f"sqlite db not found: {sqlite_path}")

    factory = get_session_factory()

    async with factory() as session, aiosqlite.connect(sqlite_path) as src:
        src.row_factory = aiosqlite.Row

        user = await get_or_create_default_user(session)

        # corpora
        existing = (await session.execute(select(models.Corpus))).scalars().all()
        if existing:
            raise SystemExit(
                f"target Postgres already has {len(existing)} corpus rows; "
                "refusing to migrate to avoid duplicates"
            )

        cur = await src.execute("SELECT * FROM corpus")
        corpus_id_map: dict[str, uuid.UUID] = {}
        async for row in cur:
            new_id = uuid.uuid4()
            corpus_id_map[row["id"]] = new_id
            session.add(models.Corpus(
                id=new_id,
                user_id=user.id,
                name=row["name"],
                tags=row["tags"] or "",
                collection_name=row["collection_name"],
                status=row["status"] or "empty",
            ))

        # documents
        cur = await src.execute("SELECT * FROM document")
        async for row in cur:
            corpus_uuid = corpus_id_map.get(row["corpus_id"])
            if not corpus_uuid:
                continue
            session.add(models.Document(
                id=uuid.uuid4(),
                corpus_id=corpus_uuid,
                filename=row["filename"],
                sha256=row["sha256"],
                file_type=row["file_type"],
                chunks=row["chunks"] or 0,
                status=row["status"] or "pending",
                author=row["author"] or "",
                title=row["title"] or "",
                year=row["year"] or 0,
                publisher=row["publisher"] or "",
                error=row["error"],
            ))

        # conversations
        cur = await src.execute("SELECT * FROM conversation")
        conv_id_map: dict[str, uuid.UUID] = {}
        async for row in cur:
            corpus_uuid = corpus_id_map.get(row["corpus_id"])
            if not corpus_uuid:
                continue
            new_id = uuid.uuid4()
            conv_id_map[row["id"]] = new_id
            session.add(models.Conversation(
                id=new_id,
                corpus_id=corpus_uuid,
                title=row["title"],
                mode=row["mode"] or "general",
                citation_style=row["citation_style"] or "chicago",
            ))

        # messages
        cur = await src.execute("SELECT * FROM message")
        msg_count = 0
        async for row in cur:
            conv_uuid = conv_id_map.get(row["conversation_id"])
            if not conv_uuid:
                continue
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            session.add(models.Message(
                id=uuid.uuid4(),
                conversation_id=conv_uuid,
                role=row["role"],
                content=row["content"],
                metadata_json=metadata,
            ))
            msg_count += 1

        await session.commit()
        print(
            f"migrated: {len(corpus_id_map)} corpora, "
            f"{len(conv_id_map)} conversations, "
            f"{msg_count} messages"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <sqlite_path>")
    asyncio.run(migrate(sys.argv[1]))
