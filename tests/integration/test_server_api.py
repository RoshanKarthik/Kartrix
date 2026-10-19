"""The HTTP API against Postgres, with a stand-in worker pool (no agent processes, no models):
accounts and tokens, per-user isolation of every route, workspaces, runs, the SSE feed and its
replay, approvals, plan reviews and stop; plus per-user long-term memories."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from kartrix.client.api import KartrixClient
from kartrix.config import settings
from kartrix.server.app import create_app
from kartrix.server.protocol import RunOp
from kartrix.server.workers import RunFeed, WorkerPool

pytestmark = pytest.mark.usefixtures("db")


class FakeWorker:
    def __init__(self, workspace_id: uuid.UUID) -> None:
        self.workspace_id = workspace_id
        self.feed: RunFeed | None = None
        self.sent: list[Any] = []
        self.exited = False
        self.last_active = 0.0

        class _Ready:
            @staticmethod
            def done() -> bool:
                return True

        self.ready = _Ready()

    @property
    def busy(self) -> bool:
        return self.feed is not None and not self.feed.finished

    def send(self, op: Any) -> None:
        self.sent.append(op)

    async def shutdown(self, grace: float = 0) -> None:
        pass


class FakePool(WorkerPool):
    """Runs start, but only the test drives them: ``emit`` worker messages, ``finish_run``."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)
        self.ops: list[RunOp] = []

    async def start_run(self, *, user_id: Any, workspace_id: Any, root: Path, run_id: Any, op: RunOp) -> RunFeed:
        worker = self.workers.get(workspace_id)
        if worker is None:
            worker = FakeWorker(workspace_id)
            self.workers[workspace_id] = worker  # type: ignore[assignment]
        feed = RunFeed(run_id, workspace_id)
        self.feeds[run_id] = feed
        worker.feed = feed
        self.ops.append(op)
        return feed

    def emit(self, run_id: str, kind: str, data: dict[str, Any]) -> None:
        feed = self.feeds[uuid.UUID(run_id)]
        if kind != "event":
            feed.pending[data["request_id"]] = kind.removesuffix("_needed")
        self.store(feed, feed.publish(kind, data, persist=data.get("type") != "assistant_delta"))

    def finish_run(self, run_id: str, **message: Any) -> None:
        feed = self.feeds[uuid.UUID(run_id)]
        self.finish(self.workers[feed.workspace_id], feed, message)  # type: ignore[arg-type]

    async def flush(self) -> None:
        assert self._writes is not None
        await self._writes.join()


@pytest.fixture
async def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[FakePool, httpx.AsyncClient]]:
    monkeypatch.setattr(settings.server, "data_dir", str(tmp_path / "server-data"))
    pool = FakePool(tmp_path / "server-data")
    app = create_app(pool)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://kartrix.test/v1") as http:
            yield pool, http


async def _signup(http: httpx.AsyncClient, email: str) -> dict[str, str]:
    r = await http.post("/auth/signup", json={"email": email, "password": "hunter2hunter2"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


async def _workspace(http: httpx.AsyncClient, headers: dict[str, str], path: Path | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "demo"}
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 — test setup
        body["path"] = str(path)
    r = await http.post("/workspaces", json=body, headers=headers)
    assert r.status_code == 201, r.text
    data: dict[str, Any] = r.json()
    return data


def _sse(text: str) -> list[tuple[str | None, str, dict[str, Any]]]:
    import json

    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        out.append((fields.get("id"), fields["event"], json.loads(fields["data"])))
    return out


async def test_signup_login_logout(server: tuple[FakePool, httpx.AsyncClient]) -> None:
    _, http = server
    headers = await _signup(http, "Ada@Example.com")
    assert (await http.get("/me", headers=headers)).json()["email"] == "ada@example.com"
    assert (await http.post("/auth/signup", json={"email": "ada@example.com", "password": "x" * 9})).status_code == 422
    assert (
        await http.post("/auth/login", json={"email": "ada@example.com", "password": "wrong-pass"})
    ).status_code == 401
    assert (
        await http.post("/auth/login", json={"email": "nobody@example.com", "password": "x" * 9})
    ).status_code == 401
    login = await http.post("/auth/login", json={"email": "ADA@example.com", "password": "hunter2hunter2"})
    assert login.status_code == 200
    second = {"Authorization": f"Bearer {login.json()['token']}"}
    assert (await http.post("/auth/logout", headers=headers)).status_code == 204
    assert (await http.get("/me", headers=headers)).status_code == 401  # revoked
    assert (await http.get("/me", headers=second)).status_code == 200  # other tokens still work
    assert (await http.get("/me")).status_code == 401
    assert (await http.get("/me", headers={"Authorization": "Bearer kx_forged"})).status_code == 401


async def test_signup_can_be_closed(
    server: tuple[FakePool, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, http = server
    monkeypatch.setattr(settings.server, "allow_signup", False)
    r = await http.post("/auth/signup", json={"email": "ada@example.com", "password": "hunter2hunter2"})
    assert r.status_code == 403


async def test_workspaces_are_private(server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path) -> None:
    _, http = server
    ada, bob = await _signup(http, "ada@example.com"), await _signup(http, "bob@example.com")
    ws = await _workspace(http, ada, tmp_path / "ada-repo")
    assert [w["id"] for w in (await http.get("/workspaces", headers=ada)).json()] == [ws["id"]]
    assert (await http.get("/workspaces", headers=bob)).json() == []
    assert (await http.get(f"/workspaces/{ws['id']}", headers=bob)).status_code == 404
    r = await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "hi"}, headers=bob)
    assert r.status_code == 404
    r = await http.post("/workspaces", json={"name": "x", "path": str(tmp_path / "ada-repo")}, headers=bob)
    assert r.status_code == 409  # one folder, one owner


async def test_workspace_paths(
    server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, http = server
    ada = await _signup(http, "ada@example.com")
    r = await http.post("/workspaces", json={"name": "x", "path": "relative/dir"}, headers=ada)
    assert r.status_code == 422
    r = await http.post("/workspaces", json={"name": "x", "path": str(tmp_path / "missing")}, headers=ada)
    assert r.status_code == 422
    r = await http.post("/workspaces", json={"name": "x", "path": str(tmp_path / "server-data")}, headers=ada)
    assert r.status_code == 422  # the server's own data
    monkeypatch.setattr(settings.server, "allow_local_paths", False)  # a hosted server
    (tmp_path / "elsewhere").mkdir()
    r = await http.post("/workspaces", json={"name": "x", "path": str(tmp_path / "elsewhere")}, headers=ada)
    assert r.status_code == 403
    hosted = await _workspace(http, ada)
    root = Path(hosted["root_path"])
    assert root.is_dir() and (tmp_path / "server-data").resolve() in root.parents  # noqa: ASYNC240


async def test_run_feed_approvals_and_replay(server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path) -> None:
    pool, http = server
    ada, bob = await _signup(http, "ada@example.com"), await _signup(http, "bob@example.com")
    ws = await _workspace(http, ada, tmp_path / "repo")
    r = await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "fix slugify"}, headers=ada)
    assert r.status_code == 202, r.text
    run = r.json()
    assert run["status"] == "running" and pool.ops[-1].input == "fix slugify"
    busy = await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "again"}, headers=ada)
    assert busy.status_code == 409

    pool.emit(run["id"], "event", {"type": "notice", "level": "info", "text": "working"})
    pool.emit(run["id"], "event", {"type": "assistant_delta", "text": "tok"})
    pool.emit(run["id"], "approval_needed", {"request_id": "q1", "requests": [{"command": "pytest"}]})
    approve = f"/runs/{run['id']}/approvals/q1"
    body = {"decisions": [{"type": "approve"}]}
    assert (await http.post(approve, json=body, headers=bob)).status_code == 404
    assert (await http.post(f"/runs/{run['id']}/approvals/nope", json=body, headers=ada)).status_code == 404
    assert (await http.post(approve, json=body, headers=ada)).status_code == 204
    assert (await http.post(approve, json=body, headers=ada)).status_code == 404  # answered already
    worker = pool.workers[uuid.UUID(ws["id"])]
    assert worker.sent[-1].decisions[0].type == "approve"  # type: ignore[attr-defined]

    pool.emit(run["id"], "plan_review_needed", {"request_id": "p1", "plan": {}})
    review = {"approved": False, "feedback": "fewer tasks"}
    assert (await http.post(f"/runs/{run['id']}/plan-reviews/p1", json=review, headers=ada)).status_code == 204
    assert worker.sent[-1].feedback == "fewer tasks"  # type: ignore[attr-defined]

    assert (await http.post(f"/runs/{run['id']}/stop", headers=bob)).status_code == 404
    assert (await http.post(f"/runs/{run['id']}/stop", headers=ada)).status_code == 202
    assert worker.sent[-1].op == "stop"  # type: ignore[attr-defined]

    pool.finish_run(run["id"], status="stopped", detail="stopped from the API", answer=None)
    await pool.flush()
    events = _sse((await http.get(f"/runs/{run['id']}/events", headers=ada)).text)  # replayed from Postgres
    assert [(seq, kind) for seq, kind, _ in events] == [
        ("1", "event"), ("2", "approval_needed"), ("3", "plan_review_needed"), ("4", "run_finished"),
    ]  # fmt: skip
    assert events[-1][2]["status"] == "stopped"
    tail = _sse((await http.get(f"/runs/{run['id']}/events", headers={**ada, "Last-Event-ID": "2"})).text)
    assert [seq for seq, _, _ in tail] == ["3", "4"]
    assert (await http.get(f"/runs/{run['id']}/events", headers=bob)).status_code == 404
    stored = (await http.get(f"/runs/{run['id']}", headers=ada)).json()
    assert stored["status"] == "stopped" and stored["finished_at"] is not None
    assert (await http.post(f"/runs/{run['id']}/stop", headers=ada)).status_code == 409


async def test_live_feed_through_the_client(server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path) -> None:
    pool, http = server
    ada = await _signup(http, "ada@example.com")
    ws = await _workspace(http, ada, tmp_path / "repo")
    client = KartrixClient("http://kartrix.test", ada["Authorization"].split()[1], transport=http._transport)
    run = await client.start_run(ws["id"], "ask", "hello")

    async def follow() -> list[tuple[str, int | None]]:
        return [(m.kind, m.seq) async for m in client.events(run["id"])]

    following = asyncio.create_task(follow())
    for _ in range(50):  # until the stream has subscribed to the live feed
        await asyncio.sleep(0.01)
        if pool.feeds[uuid.UUID(run["id"])].subscribers:
            break
    pool.emit(run["id"], "event", {"type": "notice", "level": "info", "text": "one"})
    pool.emit(run["id"], "event", {"type": "assistant_delta", "text": "tok"})
    pool.finish_run(run["id"], status="completed", answer="hi")
    kinds = await asyncio.wait_for(following, 10)
    assert kinds == [("event", 1), ("event", None), ("run_finished", 2)]  # the live feed has the deltas too
    await client.close()


async def test_workspace_commands(
    server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartrix.server.workers import PoolError

    pool, http = server
    ada, bob = await _signup(http, "ada@example.com"), await _signup(http, "bob@example.com")
    ws = await _workspace(http, ada, tmp_path / "repo")
    calls: list[dict[str, Any]] = []

    async def command(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs["name"] == "undo":
            raise PoolError(409, "a run is in progress in this workspace — try again when it ends")
        return {"mode": "read_only", "modes": ["read_only", "default", "auto"], "changed": True}

    real_command = pool.command
    monkeypatch.setattr(pool, "command", command)
    client = KartrixClient("http://kartrix.test", ada["Authorization"].split()[1], transport=http._transport)
    assert (await client.command(ws["id"], "mode", {"target": "read_only"}))["mode"] == "read_only"
    (call,) = calls
    assert (call["name"], call["args"], str(call["workspace_id"])) == ("mode", {"target": "read_only"}, ws["id"])
    assert str(call["root"]) == str((tmp_path / "repo").resolve()) and uuid.UUID(call["session_id"])

    r = await http.post(f"/workspaces/{ws['id']}/commands", json={"name": "undo"}, headers=ada)
    assert r.status_code == 409 and "run is in progress" in r.json()["detail"]
    r = await http.post(f"/workspaces/{ws['id']}/commands", json={"name": "mode"}, headers=bob)
    assert r.status_code == 404 and len(calls) == 2  # someone else's workspace: never reaches the pool
    r = await http.post(
        f"/workspaces/{ws['id']}/commands", json={"name": "x", "args": {"t": "a" * 60_000}}, headers=ada
    )
    assert r.status_code == 422

    monkeypatch.setattr(pool, "command", real_command)
    r = await http.post(f"/workspaces/{ws['id']}/commands", json={"name": "shell", "args": {}}, headers=ada)
    assert r.status_code == 422 and "unknown command" in r.json()["detail"]
    await client.close()


async def test_sessions_continue_and_stay_in_their_workspace(
    server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path
) -> None:
    from kartrix.memory.session import record_session

    pool, http = server
    ada = await _signup(http, "ada@example.com")
    ws = await _workspace(http, ada, tmp_path / "repo")
    other = str(uuid.uuid4())
    await record_session(other, str(tmp_path / "somewhere-else"))
    r = await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "x", "session_id": other}, headers=ada)
    assert r.status_code == 404  # a session of another folder can't be continued here
    mine = str(uuid.uuid4())
    await record_session(mine, ws["root_path"])
    run = (await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "x"}, headers=ada)).json()
    assert run["session_id"] == mine  # the workspace's latest conversation
    pool.finish_run(run["id"], status="completed")
    await pool.flush()
    fresh = (
        await http.post(f"/workspaces/{ws['id']}/runs", json={"input": "x", "new_session": True}, headers=ada)
    ).json()
    assert fresh["session_id"] != mine
    sessions = (await http.get(f"/workspaces/{ws['id']}/sessions", headers=ada)).json()
    assert [s["id"] for s in sessions] == [fresh["session_id"], mine]  # recorded when its run starts


async def test_user_memories_are_per_user(fake_embedder: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kartrix.memory import long_term
    from kartrix.server import auth

    ada, bob = (
        await auth.create_user("ada@example.com", "hunter2hunter2"),
        await auth.create_user("bob@example.com", "hunter2hunter2"),
    )
    monkeypatch.setenv("KARTRIX_USER_ID", str(ada.id))
    memory_id = await long_term.remember("prefers tabs", "preference", "user", repo_root=tmp_path)
    assert [m.content for m in await long_term.list_memories(tmp_path)] == ["prefers tabs"]
    monkeypatch.setenv("KARTRIX_USER_ID", str(bob.id))
    assert await long_term.list_memories(tmp_path) == []
    assert await long_term.recall("tabs", tmp_path) == []
    assert not await long_term.forget(memory_id)
    monkeypatch.delenv("KARTRIX_USER_ID")
    assert await long_term.list_memories(tmp_path) == []  # the local CLI doesn't see account memories either


async def test_model_catalog(server: tuple[FakePool, httpx.AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    _, http = server
    assert (await http.get("/models")).status_code == 401
    ada = await _signup(http, "ada@example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-key")
    data = (await http.get("/models", headers=ada)).json()
    assert data["models"][0]["default"] and data["models"][0]["id"] == data["default"]
    assert {t["id"] for t in data["tiers"]} >= {m["tier"] for m in data["models"]}
    own_key = [m for m in data["models"] if m["tier"] == "own_key"]
    assert own_key and all(m["available"] for m in own_key)  # a local server: the key is the user's own
    monkeypatch.setattr(settings.server, "allow_local_paths", False)  # hosted
    data = (await http.get("/models", headers=ada)).json()
    assert not any(m["available"] for m in data["models"] if m["tier"] == "own_key")


async def test_model_per_run_and_session(
    server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartrix.llm import catalog

    pool, http = server
    ada = await _signup(http, "ada@example.com")
    ws = await _workspace(http, ada, tmp_path / "repo")
    runs = f"/workspaces/{ws['id']}/runs"
    default = catalog.default_entry().id
    other = next(m.id for m in catalog.list_models() if m.available and not m.default)

    first = (await http.post(runs, json={"input": "x"}, headers=ada)).json()
    assert first["model"] == default and pool.ops[-1].model == default
    pool.finish_run(first["id"], status="completed")
    await pool.flush()

    r = await http.post(runs, json={"input": "x", "model": "no-such-model"}, headers=ada)
    assert r.status_code == 422 and "unknown model" in r.json()["detail"]
    picked = (await http.post(runs, json={"input": "x", "model": other}, headers=ada)).json()
    assert picked["model"] == other and pool.ops[-1].model == other
    assert picked["session_id"] == first["session_id"]
    pool.finish_run(picked["id"], status="completed")
    await pool.flush()

    again = (await http.post(runs, json={"input": "x"}, headers=ada)).json()
    assert again["model"] == other  # the session remembers the pick
    pool.finish_run(again["id"], status="completed")
    await pool.flush()
    sessions = (await http.get(f"/workspaces/{ws['id']}/sessions", headers=ada)).json()
    assert sessions[0]["model_id"] == other

    fresh = (await http.post(runs, json={"input": "x", "new_session": True}, headers=ada)).json()
    assert fresh["model"] == default  # a new conversation starts on the default
    pool.finish_run(fresh["id"], status="completed")
    await pool.flush()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-key")
    monkeypatch.setattr(settings.server, "allow_local_paths", False)  # hosted: the operator's key isn't yours
    r = await http.post(runs, json={"input": "x", "model": "claude-opus-5-5"}, headers=ada)
    assert r.status_code == 422 and "own key" in r.json()["detail"]


async def test_bootstrap_admin_and_admin_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.server import auth

    monkeypatch.setenv("KARTRIX_ADMIN_EMAIL", "Owner@Example.com")
    monkeypatch.setenv("KARTRIX_ADMIN_PASSWORD", "first-pass-123")
    assert await auth.ensure_bootstrap_admin() == "owner@example.com"
    assert await auth.ensure_bootstrap_admin() is None  # exists already: left alone
    monkeypatch.setenv("KARTRIX_ADMIN_PASSWORD", "changed-in-env-456")
    await auth.ensure_bootstrap_admin()
    assert await auth.authenticate("owner@example.com", "first-pass-123") is not None  # env never resets it

    user = await auth.authenticate("owner@example.com", "first-pass-123")
    assert user is not None
    token = await auth.issue_token(user.id, "cli")
    assert await auth.set_password("owner@example.com", "new-pass-789")
    assert await auth.user_for_token(token) is None  # old sessions signed out
    assert await auth.authenticate("owner@example.com", "new-pass-789") is not None
    assert not await auth.set_password("nobody@example.com", "whatever-123")
    assert [u.email for u in await auth.list_users()] == ["owner@example.com"]


async def test_cli_workspace_on_local_and_hosted_servers(
    server: tuple[FakePool, httpx.AsyncClient], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartrix.client.repl import find_workspace

    _, http = server
    ada = await _signup(http, "ada@example.com")
    client = KartrixClient("http://kartrix.test", ada["Authorization"].split()[1], transport=http._transport)
    folder = tmp_path / "myrepo"
    folder.mkdir()
    ws, on_server = await find_workspace(client, folder)
    assert not on_server and Path(ws["root_path"]) == folder.resolve()  # local server: this very folder
    monkeypatch.setattr(settings.server, "allow_local_paths", False)  # hosted
    other = tmp_path / "other"
    other.mkdir()
    hosted, on_server = await find_workspace(client, other)
    assert on_server and hosted["name"] == "other" and Path(hosted["root_path"]) != other
    again, _ = await find_workspace(client, other)
    assert again["id"] == hosted["id"]  # the same server workspace next time
    await client.close()
