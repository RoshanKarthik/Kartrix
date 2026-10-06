"""Core Postgres schema: sessions, projects, tasks, approvals, audit_log, code index.

Schema changes go through Alembic (``uv run alembic revision --autogenerate``);
never create tables with ``metadata.create_all`` outside tests.
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names so Alembic migrations are stable and reviewable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB}


def _str_enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    # Stored as VARCHAR + CHECK (not a native PG enum) so adding values is a simple migration.
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _updated_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ProjectStatus(enum.StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(enum.StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ApprovalKind(enum.StrEnum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    COMMAND = "command"
    FILE_WRITE = "file_write"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Session(Base):
    """One conversation thread; ``id`` is also the LangGraph thread id."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    projects: Mapped[list["Project"]] = relationship(back_populates="session")


class Project(Base):
    """One /plan invocation: a goal, its generated plan and the resulting tasks."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    plan: Mapped[dict[str, Any] | None]
    status: Mapped[ProjectStatus] = mapped_column(
        _str_enum(ProjectStatus, "project_status"),
        default=ProjectStatus.PLANNING,
        server_default=ProjectStatus.PLANNING.value,
        nullable=False,
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[Session | None] = relationship(back_populates="projects")
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Task.execution_order"
    )


class Task(Base):
    """A unit of work inside a project. ``key`` is the planner's id (e.g. ``task_1``),
    unique per project; ``depends_on`` holds those keys."""

    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_tasks_project_id_key"),
        Index("ix_tasks_project_id_status", "project_id", "status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        _str_enum(TaskStatus, "task_status"),
        default=TaskStatus.PENDING,
        server_default=TaskStatus.PENDING.value,
        nullable=False,
    )
    depends_on: Mapped[list[Any]] = mapped_column(default=list, server_default=text("'[]'::jsonb"))
    output_files: Mapped[list[Any]] = mapped_column(default=list, server_default=text("'[]'::jsonb"))
    acceptance_criteria: Mapped[list[Any]] = mapped_column(default=list, server_default=text("'[]'::jsonb"))
    result: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, server_default="3", nullable=False)
    execution_order: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    created_at: Mapped[datetime] = _created_at()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped[Project] = relationship(back_populates="tasks")


class Approval(Base):
    """A human-in-the-loop decision request (plan, tool call, command, file write)."""

    __tablename__ = "approvals"
    __table_args__ = (Index("ix_approvals_status_requested_at", "status", "requested_at"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sessions.id", ondelete="SET NULL"), index=True)
    project_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    kind: Mapped[ApprovalKind] = mapped_column(_str_enum(ApprovalKind, "approval_kind"), nullable=False)
    status: Mapped[ApprovalStatus] = mapped_column(
        _str_enum(ApprovalStatus, "approval_status"),
        default=ApprovalStatus.PENDING,
        server_default=ApprovalStatus.PENDING.value,
        nullable=False,
    )
    request: Mapped[dict[str, Any]] = mapped_column(nullable=False)  # what is being approved
    decision_reason: Mapped[str | None] = mapped_column(Text)
    decided_by: Mapped[str | None] = mapped_column(String(200))
    requested_at: Mapped[datetime] = _created_at()
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    """Append-only record of security-relevant actions.

    No foreign keys on purpose: audit rows must outlive the sessions/projects they
    describe. UPDATE and DELETE are blocked by a trigger (see the baseline migration).
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_session_id_ts", "session_id", "ts"),
        Index("ix_audit_log_action_ts", "action", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_id: Mapped[uuid.UUID | None]
    project_id: Mapped[uuid.UUID | None]
    task_id: Mapped[uuid.UUID | None]
    actor: Mapped[str] = mapped_column(String(32), nullable=False)  # user | agent | system
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g. tool.run_command
    target: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)  # allowed | denied | error | ...
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))


# Fixed by the column type: changing the embedding model's size needs a migration + full reindex.
# halfvec (not vector) because pgvector's HNSW index caps plain vectors at 2000 dims.
EMBEDDING_DIMS = 2048


class CodeFile(Base):
    """One indexed file of a repo. Its hash/mtime drive incremental re-indexing."""

    __tablename__ = "code_files"
    __table_args__ = (UniqueConstraint("repo_root", "path", name="uq_code_files_repo_root_path"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    repo_root: Mapped[str] = mapped_column(Text, nullable=False)  # absolute repo path
    path: Mapped[str] = mapped_column(Text, nullable=False)  # POSIX path relative to repo_root
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    chunker_version: Mapped[int] = mapped_column(Integer, nullable=False)  # bump → full re-chunk
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    indexed_at: Mapped[datetime] = _created_at()

    chunks: Mapped[list["CodeChunk"]] = relationship(back_populates="file", passive_deletes=True)


class CodeChunk(Base):
    """A function/class/text window of a file, with its embedding and full-text vector."""

    __tablename__ = "code_chunks"
    __table_args__ = (
        Index(
            "ix_code_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
        Index("ix_code_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("code_files.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Duplicated from code_files so the vector search can filter without a join.
    repo_root: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # function | class | block
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(HALFVEC(EMBEDDING_DIMS), nullable=False)
    tsv: Mapped[Any] = mapped_column(TSVECTOR, nullable=False)

    file: Mapped[CodeFile] = relationship(back_populates="chunks")
