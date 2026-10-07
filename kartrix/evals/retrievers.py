"""The retrieval modes the RAG eval compares.

- ``dense``    pgvector cosine search over the embedded chunks
- ``lexical``  Postgres full-text search (``sparse`` in ``retrieval.mode``)
- ``hybrid``   both, fused with reciprocal rank fusion (the default)
- ``search_first`` no index at all: ranks the repository's files by how well they match the question's
  words (what an agent finds with grep/glob before it reads), returning the best-matching window of
  each file. A stand-in for the search-first context of step 3.2, so the index can be judged against it.
- ``repo_map`` arrives with step 3.2 (listed so reports show it as not measured yet).
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from kartrix.context.discovery import discover_files
from kartrix.context.indexers.pg_index import split_identifiers

Chunk = dict[str, Any]
Retriever = Callable[[str, Path, int], Awaitable[list[Chunk]]]

INDEX_MODES = ("dense", "lexical", "hybrid")
ALL_MODES = (*INDEX_MODES, "search_first", "repo_map")
DEFAULT_MODES = (*INDEX_MODES, "search_first")
UNAVAILABLE = {"repo_map": "built in step 3.2"}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOP = frozenset(
    """a an and are as at be by can do does for from how i if in is it its of on or the this that to
    what when where which who why with without into than then there these those was were will would
    should could has have had not no yes all any each code file files function functions class
    method methods used use uses using get set does kartrix""".split()
)
_WINDOW = 40  # lines per search_first chunk
_MAX_BYTES = 512 * 1024


def query_terms(query: str) -> list[str]:
    """Searchable words of the question: identifiers whole and split, lower-case, no stop words."""
    words = _WORD.findall(query) + split_identifiers(query).split()
    parts = [p for w in words for p in (w, *w.split("_")) if p]
    terms = [t.lower() for t in parts if len(t) >= 3 and t.lower() not in _STOP]
    return list(dict.fromkeys(terms))


class SearchFirst:
    """File ranking by term statistics (BM25-style), computed fresh per repository."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, str] = {}
        for path in discover_files(root):
            try:
                if path.stat().st_size > _MAX_BYTES:
                    continue
                self.files[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
        self.lower = {rel: text.lower() for rel, text in self.files.items()}
        self.avg_len = (sum(len(t) for t in self.lower.values()) / len(self.lower)) if self.lower else 1.0

    def _idf(self, term: str) -> float:
        df = sum(1 for text in self.lower.values() if term in text)
        n = len(self.lower)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Chunk]:
        terms = query_terms(query)
        if not terms:
            return []
        idf = {t: self._idf(t) for t in terms}
        scores: dict[str, float] = {}
        for rel, text in self.lower.items():
            path = rel.lower()
            length_norm = 0.25 + 0.75 * len(text) / self.avg_len
            for t in terms:
                tf = text.count(t)
                if tf:
                    scores[rel] = scores.get(rel, 0.0) + idf[t] * (tf * 2.2) / (tf + 1.2 * length_norm)
                if t in path:
                    scores[rel] = scores.get(rel, 0.0) + 1.5 * idf[t]
        return [
            self._window(rel, terms, score)
            for rel, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
            if score > 0
        ]

    def _window(self, rel: str, terms: list[str], score: float) -> Chunk:
        lines = self.files[rel].splitlines()
        lower = [line.lower() for line in lines]
        best, best_hits = 0, -1
        for start in range(0, max(1, len(lines)), _WINDOW // 2):
            hits = sum(1 for line in lower[start : start + _WINDOW] for t in terms if t in line)
            if hits > best_hits:
                best, best_hits = start, hits
        end = min(len(lines), best + _WINDOW)
        return {
            "content": "\n".join(lines[best:end]),
            "source": rel,
            "name": None,
            "type": "window",
            "start_line": best + 1,
            "end_line": end,
            "score": round(score, 4),
        }


_search_first_cache: dict[Path, SearchFirst] = {}


async def _search_first(query: str, root: Path, k: int) -> list[Chunk]:
    engine = _search_first_cache.get(root)
    if engine is None:
        engine = await asyncio.to_thread(SearchFirst, root)
        _search_first_cache[root] = engine
    return await asyncio.to_thread(engine.search, query, k)


def _index_mode(mode: str) -> Retriever:
    from kartrix.context.retrievers.pg_hybrid import retrieve

    sql_mode = "sparse" if mode == "lexical" else mode

    async def run(query: str, root: Path, k: int) -> list[Chunk]:
        return await retrieve(query, k=k, repo_root=root, mode=sql_mode)  # type: ignore[arg-type]

    return run


def get_retriever(mode: str) -> Retriever:
    if mode in UNAVAILABLE:
        raise ValueError(f"retrieval mode {mode} is not available yet ({UNAVAILABLE[mode]})")
    if mode in INDEX_MODES:
        return _index_mode(mode)
    if mode == "search_first":
        return _search_first
    raise ValueError(f"unknown retrieval mode: {mode} (choose from {', '.join(ALL_MODES)})")
