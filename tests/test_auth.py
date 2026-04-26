"""Integration tests for FastAPI-Users JWT auth.

Talks to the real Postgres instance from docker-compose. Each test uses
a unique email so there's no collision/cleanup needed. Skipped if
DATABASE_URL isn't set or Postgres isn't reachable.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_PATH = str(Path(__file__).resolve().parent.parent / "services" / "api")
SERVICES_PATH = str(Path(__file__).resolve().parent.parent / "services")
sys.path.insert(0, API_PATH)
sys.path.insert(0, SERVICES_PATH)


def _postgres_reachable() -> bool:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://gutenberg:gutenberg_dev@127.0.0.1:5432/gutenberg",
    )
    try:
        import asyncpg
        # Convert SQLAlchemy URL to plain libpq form for asyncpg ping
        plain = url.replace("postgresql+asyncpg://", "postgresql://", 1)

        async def _ping():
            conn = await asyncpg.connect(plain, timeout=2)
            await conn.close()

        asyncio.run(_ping())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres not reachable — start docker compose up postgres",
)


@pytest.fixture
def client(monkeypatch):
    """Boot the real API app pointed at Postgres."""
    monkeypatch.setenv(
        "DATABASE_URL",
        os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://gutenberg:gutenberg_dev@127.0.0.1:5432/gutenberg",
        ),
    )
    monkeypatch.setenv("AUTH_JWT_SECRET", "test-secret-not-for-prod")
    monkeypatch.setenv("DB_BACKEND", "postgres")

    # Reload config + db to pick up env
    import importlib
    from core import config as cfg, db as core_db, auth as core_auth
    importlib.reload(cfg)
    importlib.reload(core_db)
    importlib.reload(core_auth)

    # Build a minimal app with only auth routes (avoids loading
    # the full ML stack for a quick auth integration test).
    from fastapi import FastAPI
    from core.auth import auth_backend, fastapi_users
    from core.schemas import UserCreate, UserRead, UserUpdate

    app = FastAPI()
    app.include_router(fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt")
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth"
    )
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate), prefix="/users"
    )

    with TestClient(app) as c:
        yield c


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


class TestAuthFlow:
    def test_register_login_me(self, client):
        email = _unique_email()
        r = client.post(
            "/auth/register",
            json={"email": email, "password": "supersecret123"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["email"] == email

        r = client.post(
            "/auth/jwt/login",
            data={"username": email, "password": "supersecret123"},
        )
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        assert token

        r = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["email"] == email

    def test_me_without_token_returns_401(self, client):
        r = client.get("/users/me")
        assert r.status_code == 401

    def test_login_with_wrong_password_fails(self, client):
        email = _unique_email()
        client.post(
            "/auth/register",
            json={"email": email, "password": "rightpass123"},
        )
        r = client.post(
            "/auth/jwt/login",
            data={"username": email, "password": "wrongpass"},
        )
        assert r.status_code == 400

    def test_register_duplicate_email_fails(self, client):
        email = _unique_email()
        client.post(
            "/auth/register",
            json={"email": email, "password": "pass12345"},
        )
        r = client.post(
            "/auth/register",
            json={"email": email, "password": "pass12345"},
        )
        assert r.status_code == 400
