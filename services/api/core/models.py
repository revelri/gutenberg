"""SQLAlchemy ORM models for the Phase 2 Postgres backend.

Mirrors the existing SQLite schema in ``core.database`` but adds:
- ``User`` (managed by fastapi-users)
- ``user_id`` foreign key on ``Corpus`` for tenant scoping
- All other tables inherit tenancy through the corpus FK chain

The legacy SQLite tables stay alive during migration; this is the
forward-compatible target schema. A one-shot migration script copies
existing rows into Postgres under a default user.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    """Authenticated user. Managed by fastapi-users.

    Inherits id (UUID), email, hashed_password, is_active, is_superuser,
    is_verified from SQLAlchemyBaseUserTableUUID.
    """

    __tablename__ = "user"

    tier: Mapped[str] = mapped_column(String(32), default="free", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    corpora: Mapped[list["Corpus"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Corpus(Base):
    __tablename__ = "corpus"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_corpus_user_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    tags: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    collection_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="empty", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User] = relationship(back_populates="corpora")
    documents: Mapped[list["Document"]] = relationship(
        back_populates="corpus", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="corpus", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "document"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("corpus.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    file_type: Mapped[str | None] = mapped_column(String(32))
    chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    author: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    year: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    publisher: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    corpus: Mapped[Corpus] = relationship(back_populates="documents")


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("corpus.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(512))
    mode: Mapped[str] = mapped_column(String(32), default="general", nullable=False)
    citation_style: Mapped[str] = mapped_column(
        String(32), default="chicago", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    corpus: Mapped[Corpus] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "message"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
