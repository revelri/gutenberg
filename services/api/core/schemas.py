"""Pydantic request/response schemas for auth + tenant-scoped resources.

User schemas live here to keep them out of the SQLAlchemy models module.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi_users import schemas
from pydantic import BaseModel, ConfigDict, Field


# ── User schemas (fastapi-users) ────────────────────────────────────────

class UserRead(schemas.BaseUser[uuid.UUID]):
    tier: str = "free"
    created_at: datetime | None = None


class UserCreate(schemas.BaseUserCreate):
    pass


class UserUpdate(schemas.BaseUserUpdate):
    pass


# ── Resource schemas (placeholders for Phase 2.4 router rewrite) ────────

class CorpusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    tags: str
    collection_name: str
    status: str
    created_at: datetime


class CorpusCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    tags: str = ""
