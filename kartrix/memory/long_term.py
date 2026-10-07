"""Long-term memory: what Kartrix should still know next week.

Three kinds, stored with an embedding in the ``memories`` table and recalled by relevance:

- **fact** (project scope) — conventions and knowledge about this repository ("tests run with
  `uv run pytest`", "API handlers live in app/api/routes").
- **preference** (user scope, every repository) — how the user likes things done ("prefers
  pytest over unittest", "answer briefly").
- **lesson** (project scope) — episodic memory written automatically when the reviewer rejected a
  change and the coder fixed it: the mistake not to repeat.

Plus **project instructions**: a ``KARTRIX.md`` at the repository root, always loaded (within its
token budget), like a README for the agent.

Conflicts: a new memory that is nearly the same as an existing one (cosine similarity ≥
``memory.dedupe_similarity``) *replaces* it — newer wins, and the old text is logged. Staleness: a
memory that mentions a file changed after the memory was written is recalled with a "may be stale"
note. Secrets are redacted before anything is stored or embedded.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import bindparam, delete, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from kartrix.config import settings
from kartrix.context.indexers.pg_index import repo_key
from kartrix.db.engine import session_scope
from kartrix.db.models import EMBEDDING_DIMS, Memory
from kartrix.llm.factory import get_embedder
from kartrix.observability.logger import get_logger
from kartrix.security.secrets import redact
from kartrix.security.workspace import get_workspace

logger = get_logger(__name__)

Kind = Literal["fact", "preference", "lesson"]
Source = Literal["user", "agent", "review"]
PROJECT_FILE = "KARTRIX.md"
_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.[A-Za-z0-9]{1,6})(?![\w/])")


@dataclass(frozen=True)
class Recalled:
    id: str
    kind: str
    scope: str
    content: str
    similarity: float
    updated_at: datetime
    stale_files: tuple[str, ...] = ()

    def render(self) -> str:
        note = f" (may be stale: {', '.join(self.stale_files)} changed since)" if self.stale_files else ""
        return f"- [{self.kind}] {self.content}{note}"


def _root(repo_root: str | Path | None) -> Path:
    """The repository memories belong to: the given one, else the active workspace."""
    return Path(repo_root) if repo_root is not None else get_workspace().root


def _scope_for(kind: Kind) -> Literal["project", "user"]:
    return "user" if kind == "preference" else "project"


async def remember(content: str, kind: Kind, source: Source, repo_root: str | Path | None = None) -> str:
    """Store a memory (or replace a near-duplicate: newer wins). Returns its id."""
    content = redact(content.strip())
    if not content:
        raise ValueError("a memory cannot be empty")
    scope = _scope_for(kind)
    repo = repo_key(_root(repo_root)) if scope == "project" else None
    vector = await get_embedder().aembed_query(content)
    async with session_scope() as s:
        near = await _nearest(s, vector, scope, repo, kind)
        if near is not None and near[1] >= settings.memory.dedupe_similarity:
            old_id, sim, old_text = near
            await s.execute(
                update(Memory)
                .where(Memory.id == old_id)
                .values(content=content, embedding=vector, source=source, updated_at=datetime.now(UTC))
            )
            logger.info(
                "Memory replaced (newer wins)", extra={"id": str(old_id), "similarity": round(sim, 3), "old": old_text}
            )
            return str(old_id)
        row = Memory(
            id=uuid.uuid4(), scope=scope, repo_root=repo, kind=kind, content=content, source=source, embedding=vector
        )
        s.add(row)
    logger.info("Memory stored", extra={"kind": kind, "scope": scope, "source": source})
    return str(row.id)


async def _nearest(
    s: AsyncSession, vector: list[float], scope: str, repo: str | None, kind: str
) -> tuple[uuid.UUID, float, str] | None:
    stmt = text(
        "SELECT id, 1 - (embedding <=> :v) AS sim, content FROM memories "
        "WHERE scope = :scope AND kind = :kind AND repo_root IS NOT DISTINCT FROM :repo "
        "ORDER BY embedding <=> :v LIMIT 1"
    ).bindparams(bindparam("v", type_=HALFVEC(EMBEDDING_DIMS)))
    row = (await s.execute(stmt, {"v": vector, "scope": scope, "kind": kind, "repo": repo})).first()
    return (row.id, float(row.sim), row.content) if row else None


async def recall(query: str, repo_root: str | Path | None = None, k: int | None = None) -> list[Recalled]:
    """The memories most relevant to ``query`` (this repository's + the user's), best first."""
    k = k or settings.memory.recall_k
    repo = repo_key(_root(repo_root))
    vector = await get_embedder().aembed_query(query)
    stmt = text(
        "SELECT id, kind, scope, content, updated_at, 1 - (embedding <=> :v) AS sim FROM memories "
        "WHERE scope = 'user' OR repo_root = :repo ORDER BY embedding <=> :v LIMIT :k"
    ).bindparams(bindparam("v", type_=HALFVEC(EMBEDDING_DIMS)))
    async with session_scope() as s:
        rows = (await s.execute(stmt, {"v": vector, "repo": repo, "k": k})).all()
        ids = [r.id for r in rows if float(r.sim) >= settings.memory.min_similarity]
        if ids:
            await s.execute(
                update(Memory).where(Memory.id.in_(ids)).values(uses=Memory.uses + 1, last_used_at=datetime.now(UTC))
            )
    root = Path(repo)
    return [
        Recalled(
            str(r.id),
            r.kind,
            r.scope,
            r.content,
            round(float(r.sim), 3),
            r.updated_at,
            _stale(r.content, r.updated_at, root),
        )
        for r in rows
        if float(r.sim) >= settings.memory.min_similarity
    ]


def _stale(content: str, written: datetime, root: Path) -> tuple[str, ...]:
    """Files the memory mentions that changed after it was written."""
    changed = []
    for rel in dict.fromkeys(_PATH.findall(content)):
        path = root / rel
        try:
            if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime, UTC) > written:
                changed.append(rel)
        except OSError:
            continue
    return tuple(changed)


async def list_memories(repo_root: str | Path | None = None) -> list[Memory]:
    repo = repo_key(_root(repo_root))
    async with session_scope() as s:
        rows = await s.execute(
            select(Memory)
            .where(or_(Memory.scope == "user", Memory.repo_root == repo))
            .order_by(Memory.updated_at.desc())
        )
        return list(rows.scalars().all())


async def forget(memory_id: str) -> bool:
    try:
        key = uuid.UUID(memory_id)
    except ValueError:
        return False
    async with session_scope() as s:
        result = await s.execute(delete(Memory).where(Memory.id == key))
    return bool(getattr(result, "rowcount", 0))


def project_instructions(repo_root: str | Path | None = None) -> str:
    """``KARTRIX.md`` at the repository root (empty if there is none)."""
    path = _root(repo_root) / PROJECT_FILE
    try:
        return redact(path.read_text(encoding="utf-8", errors="ignore")).strip()
    except OSError:
        return ""
