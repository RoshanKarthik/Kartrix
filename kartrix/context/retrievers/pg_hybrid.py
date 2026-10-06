"""Code retrieval from Postgres: pgvector (dense) + full-text (sparse), fused with RRF.

``retrieval.mode``:
  dense  — cosine distance on the halfvec HNSW index
  sparse — ``ts_rank_cd`` over the weighted tsvector (GIN index)
  hybrid — both, each taking ``candidates`` results, merged by reciprocal rank fusion:
           score = Σ 1 / (rrf_k + rank). Rank-based, so the two score scales never need
           to be compared. Falls back to dense when the query has no searchable words.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import bindparam, text

from kartrix.config import settings
from kartrix.context.indexers.pg_index import repo_key, split_identifiers
from kartrix.db.engine import session_scope
from kartrix.db.models import EMBEDDING_DIMS
from kartrix.llm.factory import get_embedder
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

Mode = Literal["hybrid", "dense", "sparse"]

_WORD = re.compile(r"[A-Za-z0-9]+")
_MAX_TERMS = 32

_DENSE = """
dense AS (
    SELECT id, row_number() OVER (ORDER BY dist) AS rnk
    FROM (SELECT id, embedding <=> :qvec AS dist FROM code_chunks
          WHERE repo_root = :repo ORDER BY dist LIMIT :n) d
)"""

_SPARSE = """
sparse AS (
    SELECT id, row_number() OVER (ORDER BY r DESC) AS rnk
    FROM (SELECT c.id, ts_rank_cd(c.tsv, q) AS r
          FROM code_chunks c, to_tsquery('english'::regconfig, :tsq) q
          WHERE c.repo_root = :repo AND c.tsv @@ q ORDER BY r DESC LIMIT :n) s
)"""


def build_tsquery(query: str) -> str:
    """OR-query of the words in ``query`` (plus camelCase parts).

    Only ``[A-Za-z0-9]+`` tokens survive, so user input can't inject tsquery syntax;
    the 'english' config then drops stop words and stems the rest.
    """
    words = _WORD.findall(f"{query} {split_identifiers(query)}".lower())
    return " | ".join(list(dict.fromkeys(words))[:_MAX_TERMS])


def _sql(mode: Mode) -> str:
    if mode == "dense":
        ctes, join, score = [_DENSE], "dense f", "1.0 / (:rrf_k + f.rnk)"
    elif mode == "sparse":
        ctes, join, score = [_SPARSE], "sparse f", "1.0 / (:rrf_k + f.rnk)"
    else:
        ctes = [_DENSE, _SPARSE]
        join = "dense FULL OUTER JOIN sparse USING (id)"
        score = "COALESCE(1.0 / (:rrf_k + dense.rnk), 0) + COALESCE(1.0 / (:rrf_k + sparse.rnk), 0)"
    id_col = "f.id" if mode != "hybrid" else "id"
    # Only the constant fragments above are interpolated; every value is a bound parameter.
    return f"""
WITH {",".join(ctes)},
fused AS (SELECT {id_col} AS id, {score} AS score FROM {join})
SELECT c.content, c.name, c.kind, c.start_line, c.end_line, cf.path, fused.score
FROM fused
JOIN code_chunks c ON c.id = fused.id
JOIN code_files cf ON cf.id = c.file_id
ORDER BY fused.score DESC, c.id
LIMIT :k"""  # noqa: S608


async def retrieve(
    query: str,
    k: int | None = None,
    repo_root: str | Path | None = None,
    mode: Mode | None = None,
) -> list[dict]:
    """Return the top-``k`` chunks of ``repo_root`` (default: cwd) for ``query``."""
    cfg = settings.retrieval
    k = k or cfg.top_k
    mode = mode or cfg.mode
    repo = repo_key(repo_root or Path.cwd())
    tsq = build_tsquery(query)
    if not tsq:  # nothing keyword-searchable in the query
        if mode == "sparse":
            return []
        mode = "dense"

    params: dict = {"repo": repo, "n": max(cfg.candidates, k), "k": k, "rrf_k": cfg.rrf_k}
    stmt = text(_sql(mode))
    if mode != "sparse":
        params["qvec"] = await get_embedder().aembed_query(query)
        stmt = stmt.bindparams(bindparam("qvec", type_=HALFVEC(EMBEDDING_DIMS)))
    if mode != "dense":
        params["tsq"] = tsq

    logger.info("Retrieving chunks", extra={"mode": mode, "k": k, "query": query})
    async with session_scope() as s:
        if mode != "sparse":
            # The repo filter is applied after the HNSW scan; iterative scans keep going
            # until enough rows pass it (pgvector >= 0.8). Values are ints, not user input.
            await s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            await s.execute(text(f"SET LOCAL hnsw.ef_search = {int(max(40, params['n']))}"))
        rows = (await s.execute(stmt, params)).all()

    chunks = [
        {
            "content": r.content,
            "source": r.path,
            "name": r.name,
            "type": r.kind,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "score": float(r.score),
        }
        for r in rows
    ]
    logger.info("Retrieved chunks", extra={"mode": mode, "count": len(chunks)})
    return chunks
