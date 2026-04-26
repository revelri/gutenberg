"""FastAPI-Users wiring: JWT bearer auth + user manager + dependencies.

Exposes:
- ``current_active_user`` — dependency that yields the authenticated User
- ``current_active_superuser`` — same but requires superuser
- ``get_user_db`` / ``get_user_manager`` — fastapi-users plumbing
- ``fastapi_users`` — instance whose ``get_auth_router()`` etc. mount routes

The router setup is consolidated in ``main.py`` so middleware/CORS land in
one place.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import (
    SQLAlchemyAccessTokenDatabase,
    SQLAlchemyBaseAccessTokenTableUUID,
)
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.db import Base, get_session
from core.models import User

log = logging.getLogger("gutenberg.auth")


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    """Refresh-token-equivalent table for fastapi-users DB strategy.

    We use JWT bearer for the access token (stateless) and SQL-backed
    tokens for refresh. Defined here rather than in models.py to keep
    auth-only concerns colocated.
    """


async def get_user_db(session: AsyncSession = Depends(get_session)) -> AsyncGenerator:
    yield SQLAlchemyUserDatabase(session, User)


async def get_access_token_db(
    session: AsyncSession = Depends(get_session),
) -> AsyncGenerator:
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = settings.auth_jwt_secret
    verification_token_secret = settings.auth_jwt_secret

    async def on_after_register(self, user: User, request=None):
        log.info(f"user registered: {user.id} ({user.email})")

    async def on_after_forgot_password(self, user: User, token: str, request=None):
        # TODO: hook up email backend (SES/Postmark) when SaaS goes live.
        log.info(f"password reset token for {user.email}: {token}")

    async def on_after_request_verify(self, user: User, token: str, request=None):
        log.info(f"verification token for {user.email}: {token}")


async def get_user_manager(
    user_db=Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(
        secret=settings.auth_jwt_secret,
        lifetime_seconds=settings.auth_jwt_lifetime_seconds,
    )


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)
current_active_superuser = fastapi_users.current_user(active=True, superuser=True)
