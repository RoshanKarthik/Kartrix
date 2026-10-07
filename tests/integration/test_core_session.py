"""CoreSession startup (2.2): the index is built in the background (or waited for), with its state
reported, the watcher started after it, and a timing breakdown recorded."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kartrix.context import index_status
from kartrix.context.indexers.pg_index import indexed_files
from kartrix.core.events import collecting
from kartrix.core.session import CoreSession, StartOptions
from kartrix.security import permissions
from kartrix.security import workspace as ws_mod

pytestmark = pytest.mark.usefixtures("db", "fake_embedder")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    for i in range(5):
        (root / f"mod{i}.py").write_text(f"def f{i}(x):\n    return x + {i}\n")
    monkeypatch.chdir(root)
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("kartrix.core.session.get_llm", lambda: None)
    monkeypatch.setattr("kartrix.core.session.get_embedder", lambda: None)
    monkeypatch.setattr("kartrix.core.session.build_agent", lambda checkpointer, tools=(): object())
    previous = ws_mod._current
    yield root
    ws_mod._current = previous
    permissions._mode = None
    index_status.set_status("unknown")


async def _approver(requests: list[Any]) -> list[Any]:
    return []


async def test_background_index_then_watcher(ws: Path) -> None:
    opts = StartOptions(session_id=None, mcp=False, semantic_cache=False, resume_pending=False, watch=True)
    with collecting() as seen:
        core = await CoreSession.start(_approver, None, opts)
        try:
            assert core._index_task is not None  # start() returned without waiting for the index
            assert index_status.get_status().state in ("building", "ready")
            assert {"llm", "checkpoints", "sandbox", "agent", "session"} <= set(core.startup_timings)
            await core._index_task
            assert index_status.get_status().state == "ready" and core.ready_for_search
            assert len(await indexed_files(ws)) == 5 and "index" in core.startup_timings
            assert core._observer is not None  # watching starts once the index is up to date
            assert any(e.type == "notice" and e.text.startswith("Index ready") for e in seen)
            await core.reindex()  # /reindex after the background run: just runs again
        finally:
            await core.close()


async def test_wait_mode_indexes_before_returning(ws: Path) -> None:
    opts = StartOptions(mcp=False, semantic_cache=False, resume_pending=False, watch=False, index="wait")
    core = await CoreSession.start(_approver, None, opts)
    try:
        assert core._index_task is None and core._observer is None
        assert index_status.get_status().state == "ready" and len(await indexed_files(ws)) == 5
    finally:
        await core.close()


async def test_index_failure_is_reported_not_raised(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(repo: Any) -> Any:
        raise ConnectionError("embedding service down")

    monkeypatch.setattr("kartrix.core.session.index_repo", broken)
    opts = StartOptions(mcp=False, semantic_cache=False, resume_pending=False, watch=True)
    with collecting() as seen:
        core = await CoreSession.start(_approver, None, opts)
        try:
            assert core._index_task is not None
            await core._index_task
        finally:
            await core.close()
    status = index_status.get_status()
    assert status.state == "failed" and "embedding service down" in (status.detail or "")
    assert core._observer is None
    assert any(e.type == "notice" and e.level == "warning" and "Indexing failed" in e.text for e in seen)
