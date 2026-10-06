"""Incremental codebase indexing into Postgres (pgvector + full-text).

How a run works (``index_repo``):
  1. Discover files (.gitignore-aware, see ``kartrix.context.discovery``).
  2. Compare with ``code_files``: same size + mtime + embedding model + chunker version →
     skip without reading.
     Otherwise hash the content; same hash → only refresh mtime (no embedding cost).
  3. Changed/new files are parsed, embedded in batches and written; each file's old
     chunks go away with its ``code_files`` row (ON DELETE CASCADE).
  4. Rows for files that disappeared (deleted or newly ignored) are removed.

Every chunk stores a halfvec embedding (dense search) and a weighted tsvector
(name/path = A, body = B) for keyword search — see ``retrievers.pg_hybrid``.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import ColumnClause, Text, cast, delete, func, insert, literal_column, select, update

from kartrix.config import settings
from kartrix.context.discovery import RepoFilter, discover_files
from kartrix.context.indexers.code_parser import ParsedChunk, parse_file
from kartrix.db.engine import session_scope
from kartrix.db.models import EMBEDDING_DIMS, CodeChunk, CodeFile
from kartrix.llm.factory import get_embedder
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

# Bump when chunking/embedding-text logic changes: every file is then re-chunked and
# re-embedded even if its content didn't change.
# v2: tree-sitter byte offsets sliced correctly (v1 shifted chunks after non-ASCII text).
CHUNKER_VERSION = 2

# One indexing run at a time per process (startup, /reindex, watcher and /plan can overlap).
_lock = asyncio.Lock()

# Files are embedded in groups of roughly this many chunks, then written, so progress is
# committed regularly and memory stays bounded on big repos.
_GROUP_CHUNKS = 256

_TS_CONFIG: ColumnClause[str] = literal_column("'english'::regconfig")
_WEIGHT_A: ColumnClause[str] = literal_column("'A'::\"char\"")  # setweight() takes "char", which a bound param isn't
_WEIGHT_B: ColumnClause[str] = literal_column("'B'::\"char\"")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*")


class IndexConfigError(RuntimeError):
    """Raised when the embedding size in config doesn't match the database column."""


@dataclass
class IndexStats:
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    empty: int = 0  # no chunks (empty or nothing parseable)
    chunks: int = 0

    def __str__(self) -> str:
        return (
            f"added {self.added}, changed {self.changed}, deleted {self.deleted}, "
            f"unchanged {self.unchanged}, empty {self.empty} - {self.chunks} chunks embedded"
        )


@dataclass
class _Prepared:
    rel: str
    sha256: str
    size: int
    mtime_ns: int
    is_new: bool
    chunks: list[ParsedChunk] = field(default_factory=list)


def repo_key(repo_root: str | Path) -> str:
    """Canonical repo identifier stored in ``code_files.repo_root``."""
    return str(Path(repo_root).resolve())


def split_identifiers(text: str) -> str:
    """Words hidden in camelCase/PascalCase identifiers, so full-text search finds them.

    Postgres already splits snake_case on ``_``; ``parseConfigFile`` would otherwise be one token.
    """
    words = {
        part.lower()
        for ident in set(_IDENT.findall(text))
        for part in _CAMEL.split(ident)
        if part and part.lower() != ident.lower()
    }
    return " ".join(sorted(words))


def _check_dims() -> None:
    if settings.embeddings.dims != EMBEDDING_DIMS:
        raise IndexConfigError(
            f"embeddings.dims is {settings.embeddings.dims} but the code index column is "
            f"halfvec({EMBEDDING_DIMS}); changing it needs a migration and a full reindex"
        )


def _read_and_parse(root: Path, rel: str, is_new: bool, known_sha: str | None) -> _Prepared | None:
    """Hash the file; parse it only if the content changed. Returns None if unchanged."""
    path = root / rel
    data = path.read_bytes()
    st = path.stat()
    digest = hashlib.sha256(data).hexdigest()
    prep = _Prepared(rel=rel, sha256=digest, size=len(data), mtime_ns=st.st_mtime_ns, is_new=is_new)
    if digest == known_sha:
        return None
    source = data.decode("utf-8", errors="ignore").replace("\x00", "")
    try:
        prep.chunks = parse_file(str(path), source=source)
    except (SyntaxError, ValueError) as e:  # empty or unsupported file: stored with 0 chunks
        logger.info("File produced no chunks", extra={"path": rel, "reason": str(e)})
    return prep


def _embed_text(rel: str, chunk: ParsedChunk) -> str:
    # The path and symbol name give the embedding context the bare body lacks.
    return f"{rel}\n{chunk.type} {chunk.name}\n{chunk.content}"


async def _write_group(root_key: str, group: list[_Prepared], model: str) -> int:
    """Embed every chunk of ``group`` and replace those files' rows in one transaction."""
    pairs = [(p, c) for p in group for c in p.chunks]
    vectors: list[list[float]] = []
    if pairs:
        embedder = get_embedder()
        batch = settings.index.embed_batch_size
        texts = [_embed_text(p.rel, c) for p, c in pairs]
        for i in range(0, len(texts), batch):
            vectors += await embedder.aembed_documents(texts[i : i + batch])

    async with session_scope() as s:
        await s.execute(
            delete(CodeFile).where(CodeFile.repo_root == root_key, CodeFile.path.in_([p.rel for p in group]))
        )
        file_ids = {}
        for p in group:
            file_ids[p.rel] = (
                await s.execute(
                    insert(CodeFile)
                    .values(
                        repo_root=root_key,
                        path=p.rel,
                        sha256=p.sha256,
                        size=p.size,
                        mtime_ns=p.mtime_ns,
                        embedding_model=model,
                        chunker_version=CHUNKER_VERSION,
                        chunk_count=len(p.chunks),
                    )
                    .returning(CodeFile.id)
                )
            ).scalar_one()
        for (p, c), vec in zip(pairs, vectors, strict=True):
            header = f"{c.name} {p.rel.replace('/', ' ').replace('.', ' ')}"
            body = f"{c.content}\n{split_identifiers(c.content)}"
            tsv = func.setweight(func.to_tsvector(_TS_CONFIG, cast(header, Text)), _WEIGHT_A).op("||")(
                func.setweight(func.to_tsvector(_TS_CONFIG, cast(body, Text)), _WEIGHT_B)
            )
            await s.execute(
                insert(CodeChunk).values(
                    file_id=file_ids[p.rel],
                    repo_root=root_key,
                    name=c.name,
                    kind=c.type,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content.replace("\x00", ""),
                    embedding=vec,
                    tsv=tsv,
                )
            )
    return len(pairs)


async def _index(root: Path, rels: list[str], stats: IndexStats, *, prune: bool) -> IndexStats:
    root_key = str(root)
    model = settings.embeddings.model
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(
                    CodeFile.path,
                    CodeFile.sha256,
                    CodeFile.size,
                    CodeFile.mtime_ns,
                    CodeFile.embedding_model,
                    CodeFile.chunker_version,
                ).where(CodeFile.repo_root == root_key)
            )
        ).all()
    known = {r.path: r for r in rows}

    if prune:
        gone = sorted(set(known) - set(rels))
        if gone:
            async with session_scope() as s:
                await s.execute(delete(CodeFile).where(CodeFile.repo_root == root_key, CodeFile.path.in_(gone)))
            stats.deleted += len(gone)

    group: list[_Prepared] = []
    group_chunks = 0
    for rel in rels:
        k = known.get(rel)
        # Only a row written by the same model + chunker can be reused as-is.
        reusable = k if k is not None and k.embedding_model == model and k.chunker_version == CHUNKER_VERSION else None
        try:
            st = (root / rel).stat()
            if reusable is not None and reusable.size == st.st_size and reusable.mtime_ns == st.st_mtime_ns:
                stats.unchanged += 1
                continue
            known_sha = reusable.sha256 if reusable is not None else None
            prep = await asyncio.to_thread(_read_and_parse, root, rel, k is None, known_sha)
        except OSError as e:  # vanished or unreadable between discovery and now
            logger.warning("Cannot read file", extra={"path": rel, "error": str(e)})
            continue
        if prep is None:  # touched but identical content: just remember the new mtime
            async with session_scope() as s:
                await s.execute(
                    update(CodeFile)
                    .where(CodeFile.repo_root == root_key, CodeFile.path == rel)
                    .values(mtime_ns=st.st_mtime_ns)
                )
            stats.unchanged += 1
            continue
        if prep.is_new:
            stats.added += 1
        else:
            stats.changed += 1
        if not prep.chunks:
            stats.empty += 1
        group.append(prep)
        group_chunks += len(prep.chunks)
        if group_chunks >= _GROUP_CHUNKS:
            stats.chunks += await _write_group(root_key, group, model)
            group, group_chunks = [], 0
    if group:
        stats.chunks += await _write_group(root_key, group, model)
    return stats


async def index_repo(repo_root: str | Path) -> IndexStats:
    """Bring the index for ``repo_root`` up to date (safe to call repeatedly)."""
    _check_dims()
    root = Path(repo_key(repo_root))
    async with _lock:
        files = await asyncio.to_thread(discover_files, root)
        rels = [f.relative_to(root).as_posix() for f in files]
        stats = await _index(root, rels, IndexStats(), prune=True)
    logger.info("Index updated", extra={"repo": str(root), **stats.__dict__})
    return stats


async def index_file(repo_root: str | Path, path: str | Path, repo_filter: RepoFilter | None = None) -> IndexStats:
    """(Re)index one file — or drop it if it no longer exists / is now ignored."""
    _check_dims()
    root = Path(repo_key(repo_root))
    flt = repo_filter or RepoFilter.load(root)
    rel = flt.rel(path)
    if rel is None:
        return IndexStats()
    if not flt.is_indexable(path):
        await remove_file(root, path)
        return IndexStats(deleted=1)
    async with _lock:
        return await _index(root, [rel], IndexStats(), prune=False)


async def remove_file(repo_root: str | Path, path: str | Path) -> None:
    root = Path(repo_key(repo_root))
    rel = RepoFilter.load(root).rel(path)
    if rel is None:
        return
    async with _lock, session_scope() as s:
        await s.execute(delete(CodeFile).where(CodeFile.repo_root == str(root), CodeFile.path == rel))
    logger.info("Removed file from index", extra={"path": rel})


async def show_index(repo_root: str | Path, limit: int = 30) -> None:
    """Print a summary of what is indexed for ``repo_root``."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    root_key = repo_key(repo_root)
    async with session_scope() as s:
        files = (
            await s.execute(
                select(CodeFile.path, CodeFile.chunk_count, CodeFile.indexed_at)
                .where(CodeFile.repo_root == root_key)
                .order_by(CodeFile.path)
            )
        ).all()
    total = sum(f.chunk_count for f in files)
    console.print(
        f"\n[bold]Code index — {len(files)} files, {total} chunks[/bold] "
        f"[dim]({settings.embeddings.model}, halfvec({EMBEDDING_DIMS}))[/dim]\n"
    )
    table = Table("File", "Chunks", "Indexed at")
    for f in files[:limit]:
        table.add_row(f.path, str(f.chunk_count), f.indexed_at.strftime("%Y-%m-%d %H:%M"))
    console.print(table)
    if len(files) > limit:
        console.print(f"[dim]… and {len(files) - limit} more files[/dim]")
