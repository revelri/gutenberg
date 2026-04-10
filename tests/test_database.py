import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

from core.database import set_db_path, init_db, get_db


def test_init_creates_tables(tmp_path):
    async def _check():
        set_db_path(str(tmp_path / "test.db"))
        await init_db()
        db_conn = await get_db()
        try:
            cursor = await db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in await cursor.fetchall()}
            assert "corpus" in tables
            assert "document" in tables
            assert "conversation" in tables
            assert "message" in tables
            assert "ingestion_job" in tables
        finally:
            await db_conn.close()

    asyncio.run(_check())


def test_set_db_path(tmp_path):
    from core import database

    custom_path = str(tmp_path / "custom.db")
    set_db_path(custom_path)
    assert database._DB_PATH == custom_path


def test_get_db_returns_working_connection(tmp_path):
    async def _check():
        set_db_path(str(tmp_path / "test.db"))
        await init_db()
        db_conn = await get_db()
        try:
            cursor = await db_conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1
        finally:
            await db_conn.close()

    asyncio.run(_check())


def test_foreign_keys_enabled(tmp_path):
    async def _check():
        set_db_path(str(tmp_path / "test.db"))
        await init_db()
        db_conn = await get_db()
        try:
            import uuid

            fake_conv_id = str(uuid.uuid4())
            with pytest.raises(Exception):
                await db_conn.execute(
                    "INSERT INTO message (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), fake_conv_id, "user", "orphan message"),
                )
                await db_conn.commit()
        finally:
            await db_conn.close()

    asyncio.run(_check())
