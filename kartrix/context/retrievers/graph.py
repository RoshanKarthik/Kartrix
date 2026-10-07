"""Queries over the code graph (``code_edges``, see ``kartrix.context.indexers.code_graph``).

- :func:`neighbors` — GraphRAG expansion: the chunks the top search hits call, and the chunks that
  call them, so an answer sees a function together with its callers/callees.
- :func:`symbol_graph` — where a symbol is defined, who calls it and what it calls (the
  ``symbol_graph`` tool).
- :func:`repo_map` — the most-referenced functions/classes of the repository (a context section).

**Resolution.** Edges hold names, not definitions, so a call ``save()`` could mean any ``save``. A call
is linked to a definition only when it is *resolved*: the call is in the definition's own file, or the
calling file imports the definition's module or package (``from shop.pricing import …``,
``import shop.pricing``, ``from shop import pricing``, ``import './pricing'``). Text-only, no type
inference: precise enough to stop generic names (``get``, ``run``) from linking everything to everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from kartrix.context.indexers.pg_index import repo_key
from kartrix.db.engine import session_scope
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

Chunk = dict[str, Any]
_SEEDS = 3  # top hits whose neighbours are looked up
_PER_SEED = 6

# An import target as a path without extension: "shop.pricing" → shop/pricing, "./lib/a.js" → lib/a.
_NORM = r"""(CASE WHEN strpos(i.target, '/') > 0
    THEN regexp_replace(regexp_replace(i.target, '^(\.\.?/)+', ''), '\.[^./]*$', '')
    ELSE btrim(replace(i.target, '.', '/'), '/') END)"""


def _resolved(call_file: str, def_file: str, def_path: str) -> str:
    """SQL: the call in file ``call_file`` refers to the definition in ``def_file`` (path ``def_path``)."""
    return rf"""({call_file} = {def_file} OR EXISTS (
    SELECT 1 FROM code_edges i WHERE i.file_id = {call_file} AND i.kind = 'import' AND {_NORM} <> ''
    AND strpos('/' || regexp_replace({def_path}, '\.[^./]*$', '') || '/', '/' || {_NORM} || '/') > 0))"""  # noqa: S608 — constant SQL fragments only


# What the seed calls: definitions of the names it calls, resolved from the seed's file.
_CALLEES = f"""
SELECT DISTINCT ON (c.id) c.id, c.content, c.name, c.kind, c.start_line, c.end_line, cf.path,
       (cf.path = :path) AS same_file
FROM code_edges e
JOIN code_files sf ON sf.id = e.file_id
JOIN code_chunks c ON c.repo_root = e.repo_root AND c.name = e.target AND c.kind <> 'block'
JOIN code_files cf ON cf.id = c.file_id
WHERE e.repo_root = :repo AND e.kind = 'call' AND sf.path = :path AND e.source = :name AND e.target <> :name
  AND {_resolved("e.file_id", "c.file_id", "cf.path")}
ORDER BY c.id
LIMIT :n"""  # noqa: S608 — only constant SQL fragments are interpolated

# Who calls the seed: chunks whose calls of its name resolve to the seed's file.
_CALLERS = f"""
SELECT DISTINCT ON (c.id) c.id, c.content, c.name, c.kind, c.start_line, c.end_line, cf.path,
       (cf.path = :path) AS same_file
FROM code_files df
JOIN code_edges e ON e.repo_root = df.repo_root AND e.kind = 'call' AND e.target = :name AND e.source <> :name
JOIN code_chunks c ON c.file_id = e.file_id AND c.name = e.source
JOIN code_files cf ON cf.id = c.file_id
WHERE df.repo_root = :repo AND df.path = :path AND {_resolved("e.file_id", "df.id", "df.path")}
ORDER BY c.id
LIMIT :n"""  # noqa: S608

_DEFS = """
SELECT cf.path, c.kind, c.start_line, c.end_line FROM code_chunks c JOIN code_files cf ON cf.id = c.file_id
WHERE c.repo_root = :repo AND c.name = :name AND c.kind <> 'block'
ORDER BY cf.path, c.start_line LIMIT :n"""

_SYMBOL_CALLERS = f"""
SELECT DISTINCT e.source, ef.path, e.line
FROM code_chunks c
JOIN code_files cf ON cf.id = c.file_id
JOIN code_edges e ON e.repo_root = c.repo_root AND e.kind = 'call' AND e.target = c.name AND e.source <> c.name
JOIN code_files ef ON ef.id = e.file_id
WHERE c.repo_root = :repo AND c.name = :name AND c.kind <> 'block' AND {_resolved("e.file_id", "c.file_id", "cf.path")}
ORDER BY ef.path, e.line LIMIT :n"""  # noqa: S608

_SYMBOL_CALLEES = """
SELECT DISTINCT e.target FROM code_chunks c
JOIN code_edges e ON e.file_id = c.file_id AND e.kind = 'call' AND e.source = c.name AND e.target <> c.name
WHERE c.repo_root = :repo AND c.name = :name AND c.kind <> 'block'
ORDER BY e.target LIMIT :n"""

_REPO_MAP = f"""
SELECT cf.path, c.name, c.kind, c.start_line, COUNT(DISTINCT e.id) AS refs
FROM code_chunks c
JOIN code_files cf ON cf.id = c.file_id
JOIN code_edges e ON e.repo_root = c.repo_root AND e.kind = 'call' AND e.target = c.name AND e.source <> c.name
WHERE c.repo_root = :repo AND c.kind <> 'block' AND {_resolved("e.file_id", "c.file_id", "cf.path")}
GROUP BY cf.path, c.name, c.kind, c.start_line
ORDER BY refs DESC, cf.path, c.start_line
LIMIT :n"""  # noqa: S608


def _key(c: Chunk) -> tuple[Any, ...]:
    return (c["source"], c.get("name"), c["start_line"])


async def neighbors(seeds: list[Chunk], repo_root: str | Path | None = None, n: int = 2) -> list[Chunk]:
    """Up to ``n`` chunks one resolved call-edge away from the top ``seeds`` (not already among them), best
    first: connected to a higher-ranked seed, then in the seed's file, then to more seeds."""
    if n <= 0:
        return []
    repo = repo_key(repo_root or Path.cwd())
    seen = {_key(c) for c in seeds}
    found: dict[tuple[Any, ...], tuple[Chunk, list[float]]] = {}
    named = [c for c in seeds if c.get("name") and c.get("type") != "block"][:_SEEDS]
    async with session_scope() as s:
        for rank, seed in enumerate(named):
            params = {"repo": repo, "path": seed["source"], "name": seed["name"], "n": _PER_SEED}
            for sql, relation in ((_CALLEES, "called by"), (_CALLERS, "calls")):
                for r in (await s.execute(text(sql), params)).all():
                    chunk = {
                        "content": r.content, "source": r.path, "name": r.name, "type": r.kind,
                        "start_line": r.start_line, "end_line": r.end_line, "score": 0.0,
                        "via": f"{relation} {seed['name']}",
                    }  # fmt: skip
                    key = _key(chunk)
                    if key in seen:
                        continue
                    entry = found.setdefault(key, (chunk, []))
                    entry[1].append(rank - (0.5 if r.same_file else 0.0))
    ranked = sorted(found.values(), key=lambda e: (min(e[1]), -len(e[1]), e[0]["source"], e[0]["start_line"]))
    return [chunk for chunk, _ in ranked[:n]]


async def symbol_graph(name: str, repo_root: str | Path | None = None, limit: int = 20) -> dict[str, list[Any]]:
    """Definitions of ``name``, its resolved callers (who, file, line) and the names it calls."""
    params = {"repo": repo_key(repo_root or Path.cwd()), "name": name, "n": limit}
    async with session_scope() as s:
        defs = (await s.execute(text(_DEFS), params)).all()
        callers = (await s.execute(text(_SYMBOL_CALLERS), params)).all()
        callees = (await s.execute(text(_SYMBOL_CALLEES), params)).all()
    return {
        "definitions": [(r.path, r.kind, r.start_line, r.end_line) for r in defs],
        "callers": [(r.source, r.path, r.line) for r in callers],
        "callees": [r.target for r in callees],
    }


@dataclass(frozen=True)
class MapEntry:
    path: str
    name: str
    kind: str
    line: int
    refs: int  # resolved call sites in other functions


async def repo_map(repo_root: str | Path | None = None, limit: int = 60) -> list[MapEntry]:
    """The repository's functions/classes ranked by how often they are called (resolved call sites)."""
    async with session_scope() as s:
        rows = (await s.execute(text(_REPO_MAP), {"repo": repo_key(repo_root or Path.cwd()), "n": limit})).all()
    return [MapEntry(r.path, r.name, r.kind, r.start_line, int(r.refs)) for r in rows]
