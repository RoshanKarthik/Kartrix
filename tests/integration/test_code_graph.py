"""The code graph in Postgres: edges written while indexing (and backfilled), graph expansion of search
results, the symbol_graph tool, the repo map and the ``graph`` eval mode — with the fake embedder."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import delete, func, select

from kartrix.agent.tools import search_codebase, symbol_graph
from kartrix.config import settings
from kartrix.context.indexers.pg_index import index_repo
from kartrix.context.retrievers import graph
from kartrix.context.retrievers.pg_hybrid import retrieve
from kartrix.db.engine import session_scope
from kartrix.db.models import CodeEdge
from kartrix.evals.retrievers import get_retriever
from tests.fakes import HashingEmbeddings

pytestmark = pytest.mark.usefixtures("db")


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def app_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _w(
        repo,
        "shop/pricing.py",
        "def apply_discount(total, code):\n    '''Reduce the total by a discount code.'''\n"
        "    return total - lookup_discount(code)\n\n\n"
        "def lookup_discount(code):\n    return DISCOUNTS.get(code, 0)\n",
    )
    _w(
        repo,
        "shop/checkout.py",
        "from shop.pricing import apply_discount\n\n\n"
        "def checkout(cart):\n    total = sum(i.price for i in cart.items)\n"
        "    return apply_discount(total, cart.code)\n",
    )
    _w(
        repo,
        "shop/report.py",
        "from shop import pricing\n\n\n"
        "def monthly_report(orders):\n    return [pricing.apply_discount(o.total, None) for o in orders]\n",
    )
    # calls a same-named function it never imports: must not be linked to shop/pricing.py
    _w(repo, "legacy/old.py", "def old_total(x):\n    return apply_discount(x, None)\n")
    _w(repo, "README.md", "# Shop\nA tiny shop.\n")
    monkeypatch.chdir(repo)
    return repo


async def _edge_count(repo: Path) -> int:
    async with session_scope() as s:
        return int((await s.execute(select(func.count()).select_from(CodeEdge))).scalar_one())


async def test_indexing_writes_edges_and_changes_replace_them(app_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(app_repo)
    g = await graph.symbol_graph("apply_discount", app_repo)
    assert g["definitions"] == [("shop/pricing.py", "function", 1, 3)]
    assert {(src, path) for src, path, _ in g["callers"]} == {
        ("checkout", "shop/checkout.py"),
        ("monthly_report", "shop/report.py"),
    }
    assert g["callees"] == ["lookup_discount"]

    _w(app_repo, "shop/report.py", "def monthly_report(orders):\n    return len(orders)\n")
    await index_repo(app_repo)  # the changed file's old edges went with its row (ON DELETE CASCADE)
    g = await graph.symbol_graph("apply_discount", app_repo)
    assert [src for src, _, _ in g["callers"]] == ["checkout"]


async def test_repo_indexed_before_the_graph_gets_a_backfill(app_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(app_repo)
    edges = await _edge_count(app_repo)
    async with session_scope() as s:
        await s.execute(delete(CodeEdge))
    calls = fake_embedder.calls
    await index_repo(app_repo)  # nothing changed: no embedding calls, but the graph is rebuilt
    assert fake_embedder.calls == calls and await _edge_count(app_repo) == edges > 0


async def test_search_is_expanded_with_callers_and_callees(
    app_repo: Path, fake_embedder: HashingEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    await index_repo(app_repo)
    seeds = await retrieve("discount code total", k=1, repo_root=app_repo)
    assert seeds[0]["name"] == "apply_discount"
    related = await graph.neighbors(seeds, app_repo, n=3)
    assert {(c["name"], c["via"]) for c in related} == {
        ("lookup_discount", "called by apply_discount"),
        ("checkout", "calls apply_discount"),
        ("monthly_report", "calls apply_discount"),
    }
    assert related[0]["name"] == "lookup_discount"  # same file as the seed ranks first
    assert await graph.neighbors(seeds, app_repo, n=0) == []

    monkeypatch.setattr(settings.retrieval, "top_k", 1)  # neighbours already in the results aren't repeated
    out = await search_codebase.ainvoke({"query": "discount code total"})
    assert "Related via the code graph" in out


async def test_symbol_graph_tool_and_repo_map(app_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(app_repo)
    out = await symbol_graph.ainvoke({"name": "pricing.apply_discount"})
    assert "defined: shop/pricing.py:1-3 (function)" in out
    assert "called by: checkout at shop/checkout.py:6" in out and "calls: lookup_discount" in out
    assert "not in the code graph" in await symbol_graph.ainvoke({"name": "nothing_here"})
    top = (await graph.repo_map(app_repo))[0]
    assert (top.name, top.refs) == ("apply_discount", 2)


async def test_graph_eval_mode_keeps_k(app_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(app_repo)
    chunks = await get_retriever("graph")("discount code total", app_repo, 3)
    assert len(chunks) == 3 and chunks[0]["name"] == "apply_discount"
    assert any("via" in c for c in chunks)
