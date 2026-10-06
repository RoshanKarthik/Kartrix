"""LangGraph checkpointer on Postgres, using the app's async SQLAlchemy/asyncpg engine.

Why not ``langgraph-checkpoint-postgres``: it is built on psycopg 3, whose async mode
can't run on Windows' default (Proactor) event loop, and switching loops breaks the
stdio MCP servers. This saver follows the same contract as LangGraph's SQLite saver:
whole checkpoints are stored as serializer blobs, pending writes per task, versions
are ``"<n:032>.<random>"`` strings. Tables are managed by Alembic (``checkpoints``,
``checkpoint_writes``).

Sync methods exist only for callers on other threads; they hop onto the event loop
the saver was created on, like the SQLite saver does.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.base import Checkpoint as LGCheckpoint
from sqlalchemy import cast, delete, select
from sqlalchemy.dialects.postgresql import JSONB, insert

from kartrix.db.engine import session_scope
from kartrix.db.models import Checkpoint, CheckpointWrite
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


def _jsonb_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    # JSONB rejects \u0000 anywhere; get_checkpoint_metadata only strips it from top-level strings.
    return json.loads(json.dumps(metadata, ensure_ascii=False, default=str).replace("\\u0000", ""))


def _cfg(thread_id: str, ns: str, checkpoint_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}}


class PgCheckpointSaver(BaseCheckpointSaver[str]):
    """Async LangGraph checkpoint saver backed by the ``checkpoints`` tables."""

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self.loop = asyncio.get_running_loop()

    # ── helpers ────────────────────────────────────────────────────────

    async def _writes(self, session, thread_id: str, ns: str, checkpoint_id: str) -> list[tuple[str, str, Any]]:
        rows = (
            await session.execute(
                select(CheckpointWrite.task_id, CheckpointWrite.channel, CheckpointWrite.type, CheckpointWrite.value)
                .where(
                    CheckpointWrite.thread_id == thread_id,
                    CheckpointWrite.checkpoint_ns == ns,
                    CheckpointWrite.checkpoint_id == checkpoint_id,
                )
                .order_by(CheckpointWrite.task_id, CheckpointWrite.idx)
            )
        ).all()
        return [(r.task_id, r.channel, self.serde.loads_typed((r.type, r.value))) for r in rows]

    async def _to_tuple(self, session, row: Checkpoint) -> CheckpointTuple:
        return CheckpointTuple(
            _cfg(row.thread_id, row.checkpoint_ns, row.checkpoint_id),
            self.serde.loads_typed((row.type, row.checkpoint)),
            row.metadata_ or {},
            _cfg(row.thread_id, row.checkpoint_ns, row.parent_checkpoint_id) if row.parent_checkpoint_id else None,
            await self._writes(session, row.thread_id, row.checkpoint_ns, row.checkpoint_id),
        )

    def _bridge(self, coro):
        """Run ``coro`` on the saver's loop from another thread (sync API)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                coro.close()
                raise asyncio.InvalidStateError(
                    "Synchronous checkpointer calls are only allowed from another thread; "
                    "use the async API (e.g. graph.ainvoke) on the event loop."
                )
        except RuntimeError:
            pass  # no running loop in this thread: fine
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    # ── async API ──────────────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        ns = config["configurable"].get("checkpoint_ns", "")
        stmt = select(Checkpoint).where(Checkpoint.thread_id == thread_id, Checkpoint.checkpoint_ns == ns)
        if checkpoint_id := get_checkpoint_id(config):
            stmt = stmt.where(Checkpoint.checkpoint_id == checkpoint_id)
        else:
            stmt = stmt.order_by(Checkpoint.checkpoint_id.desc()).limit(1)
        async with session_scope() as s:
            row = (await s.execute(stmt)).scalar_one_or_none()
            return await self._to_tuple(s, row) if row is not None else None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        stmt = select(Checkpoint)
        if config is not None:
            stmt = stmt.where(Checkpoint.thread_id == str(config["configurable"]["thread_id"]))
            ns = config["configurable"].get("checkpoint_ns")
            if ns is not None:
                stmt = stmt.where(Checkpoint.checkpoint_ns == ns)
            if checkpoint_id := get_checkpoint_id(config):
                stmt = stmt.where(Checkpoint.checkpoint_id == checkpoint_id)
        if filter:
            # Containment on JSONB; values are bound parameters, so keys need no escaping.
            stmt = stmt.where(Checkpoint.metadata_.op("@>")(cast(_jsonb_safe(filter), JSONB)))
        if before is not None and (before_id := get_checkpoint_id(before)):
            stmt = stmt.where(Checkpoint.checkpoint_id < before_id)
        stmt = stmt.order_by(Checkpoint.checkpoint_id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        async with session_scope() as s:
            rows = (await s.execute(stmt)).scalars().all()
            tuples = [await self._to_tuple(s, r) for r in rows]
        for t in tuples:
            yield t

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: LGCheckpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = str(config["configurable"]["thread_id"])
        ns = config["configurable"].get("checkpoint_ns", "")
        type_, blob = self.serde.dumps_typed(checkpoint)
        values = {
            "thread_id": thread_id,
            "checkpoint_ns": ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": blob,
            "metadata": _jsonb_safe(get_checkpoint_metadata(config, metadata)),
        }
        stmt = insert(Checkpoint.__table__).values(values)
        stmt = stmt.on_conflict_do_update(
            constraint="pk_checkpoints",
            set_={k: stmt.excluded[k] for k in ("parent_checkpoint_id", "type", "checkpoint", "metadata")},
        )
        async with session_scope() as s:
            await s.execute(stmt)
        return _cfg(thread_id, ns, checkpoint["id"])

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        conf = config["configurable"]
        rows = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            rows.append(
                {
                    "thread_id": str(conf["thread_id"]),
                    "checkpoint_ns": str(conf.get("checkpoint_ns", "")),
                    "checkpoint_id": str(conf["checkpoint_id"]),
                    "task_id": task_id,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                    "channel": channel,
                    "type": type_,
                    "value": blob,
                    "task_path": task_path,
                }
            )
        stmt = insert(CheckpointWrite.__table__).values(rows)
        # Special writes (errors, interrupts, ...) replace earlier ones; regular writes are
        # write-once, matching LangGraph's reference savers.
        if all(w[0] in WRITES_IDX_MAP for w in writes):
            stmt = stmt.on_conflict_do_update(
                constraint="pk_checkpoint_writes",
                set_={k: stmt.excluded[k] for k in ("channel", "type", "value", "task_path")},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(constraint="pk_checkpoint_writes")
        async with session_scope() as s:
            await s.execute(stmt)

    async def adelete_thread(self, thread_id: str) -> None:
        async with session_scope() as s:
            await s.execute(delete(Checkpoint).where(Checkpoint.thread_id == str(thread_id)))
            await s.execute(delete(CheckpointWrite).where(CheckpointWrite.thread_id == str(thread_id)))
        logger.info("Deleted checkpoint thread", extra={"thread_id": str(thread_id)})

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    # ── sync API (other threads only) ──────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._bridge(self.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        async def _collect() -> list[CheckpointTuple]:
            return [t async for t in self.alist(config, filter=filter, before=before, limit=limit)]

        yield from self._bridge(_collect())

    def put(
        self,
        config: RunnableConfig,
        checkpoint: LGCheckpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._bridge(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        return self._bridge(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        return self._bridge(self.adelete_thread(thread_id))
