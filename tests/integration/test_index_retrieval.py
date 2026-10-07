import os
from pathlib import Path

import pytest

from kartrix.context.indexers.pg_index import index_file, index_repo, remove_file
from kartrix.context.retrievers.pg_hybrid import retrieve
from tests.fakes import HashingEmbeddings

pytestmark = pytest.mark.usefixtures("db")


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def code_repo(repo: Path) -> Path:
    _w(repo, ".gitignore", "*.log\n")
    _w(
        repo,
        "billing/invoice.py",
        "def computeInvoiceTotal(lines):\n    '''Sum invoice line amounts.'''\n    return sum(l.amount for l in lines)\n",
    )
    _w(
        repo,
        "auth/login.py",
        "def verify_password(user, password):\n    '''Check a password hash.'''\n    return user.hash == password\n",
    )
    _w(repo, "docs/guide.md", "# Guide\nHow to deploy the service with docker compose.\n")
    _w(repo, "debug.log", "def leaked():\n    pass\n")
    return repo


async def test_index_then_incremental_updates(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    s1 = await index_repo(code_repo)
    assert (s1.added, s1.chunks) == (3, 3)  # .gitignore'd log excluded

    calls = fake_embedder.calls
    s2 = await index_repo(code_repo)
    assert (s2.unchanged, fake_embedder.calls) == (3, calls)  # nothing re-embedded

    p = code_repo / "auth/login.py"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # touched, same content
    s3 = await index_repo(code_repo)
    assert (s3.changed, fake_embedder.calls) == (0, calls)

    _w(code_repo, "auth/login.py", "def verify_password(u, p):\n    return False\n\n\ndef logout():\n    pass\n")
    (code_repo / "docs/guide.md").unlink()
    s4 = await index_repo(code_repo)
    assert (s4.changed, s4.deleted, s4.chunks) == (1, 1, 2)


async def test_newly_ignored_files_are_pruned(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(code_repo)
    _w(code_repo, ".gitignore", "*.log\ndocs/\n")
    assert (await index_repo(code_repo)).deleted == 1
    assert all(r["source"] != "docs/guide.md" for r in await retrieve("deploy docker", repo_root=code_repo))


async def test_single_file_index_and_remove(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(code_repo)
    _w(code_repo, "billing/refund.py", "def issue_refund(order):\n    return order.total\n")
    assert (await index_file(code_repo, code_repo / "billing/refund.py")).added == 1
    assert (await index_file(code_repo, code_repo / "debug.log")).added == 0  # ignored
    await remove_file(code_repo, code_repo / "billing/refund.py")
    assert all(r["source"] != "billing/refund.py" for r in await retrieve("refund order", repo_root=code_repo))


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
async def test_retrieval_modes_find_the_right_chunk(
    code_repo: Path, fake_embedder: HashingEmbeddings, mode: str
) -> None:
    await index_repo(code_repo)
    hits = await retrieve("verify password", repo_root=code_repo, mode=mode)  # type: ignore[arg-type]
    assert hits[0]["source"] == "auth/login.py" and hits[0]["name"] == "verify_password"
    assert hits[0]["start_line"] == 1


async def test_sparse_finds_camel_case_words(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(code_repo)
    hits = await retrieve("invoice total", repo_root=code_repo, mode="sparse")
    assert hits[0]["name"] == "computeInvoiceTotal"


async def test_hostile_query_is_harmless(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    await index_repo(code_repo)
    for q in ["'); DROP TABLE code_chunks; --", "a & !b | (c:*)", "?? !!"]:
        await retrieve(q, repo_root=code_repo)
    assert await retrieve("verify password", repo_root=code_repo)  # table still there


async def test_repos_are_isolated(code_repo: Path, tmp_path: Path, fake_embedder: HashingEmbeddings) -> None:
    other = tmp_path / "other"
    _w(other, "x.py", "def verify_password_elsewhere():\n    pass\n")
    await index_repo(code_repo)
    await index_repo(other)
    assert {r["source"] for r in await retrieve("verify password", repo_root=other)} == {"x.py"}


async def test_reranker_reorders_the_fused_candidates(code_repo: Path, fake_embedder: HashingEmbeddings) -> None:
    """Stage 2 reads all first-stage candidates (not just top k) and decides the final order."""
    from kartrix.context.rerank import set_reranker

    seen: list[int] = []

    def prefer_docs(query: str, passages: list[str]) -> list[float]:
        seen.append(len(passages))
        return [1.0 if p.startswith("docs/") else 0.0 for p in passages]

    await index_repo(code_repo)
    set_reranker(lambda: prefer_docs)
    hits = await retrieve("verify password", k=1, repo_root=code_repo)
    assert seen == [3]  # every candidate in the repo, though k = 1
    assert [h["source"] for h in hits] == ["docs/guide.md"]
    plain = await retrieve("verify password", k=1, repo_root=code_repo, rerank=False)
    assert plain[0]["source"] == "auth/login.py" and seen == [3]
