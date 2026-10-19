"""The HTTP API (FastAPI). Every route but sign-up, login and health needs a bearer token, and
everything a user can reach — workspaces, runs, their feeds and questions — is checked to be theirs;
someone else's id answers 404, exactly like an id that doesn't exist.

Routes (all JSON, under ``/v1``):

- ``POST /auth/signup`` · ``POST /auth/login`` → ``{token, user}`` · ``POST /auth/logout`` · ``GET /me``
- ``GET|POST /workspaces`` · ``GET /workspaces/{id}`` · ``GET /workspaces/{id}/sessions`` ·
  ``GET /workspaces/{id}/runs``
- ``POST /workspaces/{id}/runs`` → the run (202); one run at a time per workspace (409)
- ``GET /runs/{id}`` · ``GET /runs/{id}/events`` (Server-Sent Events; ``Last-Event-ID`` or ``?after=`` replays)
- ``POST /runs/{id}/approvals/{request_id}`` · ``POST /runs/{id}/plan-reviews/{request_id}`` ·
  ``POST /runs/{id}/stop``
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from kartrix.config import settings
from kartrix.db.engine import dispose_engine, session_scope
from kartrix.db.models import Run, RunEvent, Session, User, Workspace
from kartrix.memory.session import InvalidSessionIdError, validate_session_id
from kartrix.observability.logger import get_logger
from kartrix.paths import user_data_dir
from kartrix.server import auth
from kartrix.server.protocol import Decision, PlanReviewOp, RunOp
from kartrix.server.workers import FeedItem, PoolError, WorkerPool

logger = get_logger(__name__)

HEARTBEAT_SECONDS = 15.0
MAX_INPUT = 50_000


# ── request / response models ─────────────────────────────────────────


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=auth.MAX_PASSWORD)
    token_name: str = Field("cli", max_length=100)


class UserOut(_Out):
    id: uuid.UUID
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    token: str
    user: UserOut


class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    path: str | None = Field(None, max_length=4096)  # an existing folder (local servers only)


class WorkspaceOut(_Out):
    id: uuid.UUID
    name: str
    root_path: str
    created_at: datetime


class SessionOut(_Out):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class RunIn(BaseModel):
    kind: Literal["ask", "plan"] = "ask"
    input: str = Field(min_length=1, max_length=MAX_INPUT)
    session_id: str | None = None  # None: the workspace's latest session (or a new one)
    new_session: bool = False


class RunOut(_Out):
    id: uuid.UUID
    workspace_id: uuid.UUID
    session_id: uuid.UUID
    kind: str
    input: str
    status: str
    detail: str | None
    answer: str | None
    usage: dict[str, Any] | None
    created_at: datetime
    finished_at: datetime | None


class ApprovalIn(BaseModel):
    decisions: list[Decision] = Field(min_length=1, max_length=50)


class PlanReviewIn(BaseModel):
    approved: bool
    plan: dict[str, Any] | None = None
    feedback: str = Field("", max_length=10_000)


class ServerInfo(BaseModel):
    name: str = "kartrix"
    allow_signup: bool
    allow_local_paths: bool


# ── app ───────────────────────────────────────────────────────────────


def server_data_dir() -> Path:
    path = Path(settings.server.data_dir) if settings.server.data_dir else user_data_dir() / "server"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def create_app(pool: WorkerPool | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        workers = pool or WorkerPool(server_data_dir())
        app.state.pool = workers
        async with session_scope() as s:  # runs cut off by a restart can't finish any more
            await s.execute(
                update(Run).where(Run.status == "running").values(status="error", detail="the server restarted")
            )
        await workers.open()
        try:
            yield
        finally:
            await workers.close()
            await dispose_engine()

    app = FastAPI(title="Kartrix", version="1", lifespan=lifespan, docs_url="/docs", redoc_url=None)

    @app.exception_handler(PoolError)
    async def pool_error(request: Request, exc: PoolError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    app.include_router(_routes())
    return app


def _pool(request: Request) -> WorkerPool:
    pool: WorkerPool = request.app.state.pool
    return pool


Pool = Annotated[WorkerPool, Depends(_pool)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not found")


async def _workspace(user: User, workspace_id: uuid.UUID) -> Workspace:
    async with session_scope() as s:
        ws = (
            await s.execute(select(Workspace).where(Workspace.id == workspace_id, Workspace.user_id == user.id))
        ).scalar_one_or_none()
    if ws is None:
        raise _not_found("workspace")
    return ws


async def _run(user: User, run_id: uuid.UUID) -> Run:
    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id, Run.user_id == user.id))).scalar_one_or_none()
    if run is None:
        raise _not_found("run")
    return run


def _local_root(path: str) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "path must be an existing absolute folder")
    root = root.resolve()
    data = server_data_dir()
    if root == data or data in root.parents or root in data.parents:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "that folder holds the server's own data")
    return root


def _hosted_root(user_id: uuid.UUID, workspace_id: uuid.UUID) -> Path:
    root = server_data_dir() / "workspaces" / str(user_id) / str(workspace_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


async def _session_for(ws: Workspace, body: RunIn) -> str:
    """The session the run continues: the one asked for (it must belong to this workspace), the
    workspace's latest, or a new one."""
    async with session_scope() as s:
        if body.session_id is not None:
            try:
                session_id = validate_session_id(body.session_id)
            except InvalidSessionIdError:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid session id") from None
            row = (await s.execute(select(Session).where(Session.id == uuid.UUID(session_id)))).scalar_one_or_none()
            if row is not None and not _same_path(row.repo_path, ws.root_path):
                raise _not_found("session")
            return session_id
        if not body.new_session:
            latest = (
                await s.execute(
                    select(Session.id)
                    .where(Session.repo_path == ws.root_path)
                    .order_by(Session.updated_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if latest is not None:
                return str(latest)
    return str(uuid.uuid4())


def _sse(item: FeedItem) -> bytes:
    head = f"id: {item.seq}\n" if item.seq is not None else ""
    data = json.dumps(item.data, ensure_ascii=False, default=str)
    return f"{head}event: {item.kind}\ndata: {data}\n\n".encode()


def _routes() -> Any:
    from fastapi import APIRouter

    r = APIRouter(prefix="/v1")

    @r.get("/health")
    async def health() -> ServerInfo:
        return ServerInfo(
            allow_signup=settings.server.allow_signup, allow_local_paths=settings.server.allow_local_paths
        )

    # ── accounts ──

    @r.post("/auth/signup", status_code=status.HTTP_201_CREATED)
    async def signup(body: Credentials) -> TokenOut:
        if not settings.server.allow_signup:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "sign-up is closed on this server")
        try:
            user = await auth.create_user(body.email, body.password)
        except ValueError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from None
        return TokenOut(token=await auth.issue_token(user.id, body.token_name), user=UserOut.model_validate(user))

    @r.post("/auth/login")
    async def login(body: Credentials) -> TokenOut:
        user = await auth.authenticate(body.email, body.password)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "wrong email or password")
        return TokenOut(token=await auth.issue_token(user.id, body.token_name), user=UserOut.model_validate(user))

    @r.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    async def logout(token: Annotated[str, Depends(auth.bearer_token)]) -> None:
        await auth.revoke_token(token)

    @r.get("/me")
    async def me(user: auth.CurrentUser) -> UserOut:
        return UserOut.model_validate(user)

    # ── workspaces ──

    @r.get("/workspaces")
    async def list_workspaces(user: auth.CurrentUser) -> list[WorkspaceOut]:
        async with session_scope() as s:
            rows = await s.execute(select(Workspace).where(Workspace.user_id == user.id).order_by(Workspace.created_at))
            return [WorkspaceOut.model_validate(w) for w in rows.scalars()]

    @r.post("/workspaces", status_code=status.HTTP_201_CREATED)
    async def create_workspace(user: auth.CurrentUser, body: WorkspaceIn) -> WorkspaceOut:
        workspace_id = uuid.uuid4()
        if body.path is not None:
            if not settings.server.allow_local_paths:
                raise HTTPException(status.HTTP_403_FORBIDDEN, "this server does not work on folders of its machine")
            root = await asyncio.to_thread(_local_root, body.path)
        else:
            root = await asyncio.to_thread(_hosted_root, user.id, workspace_id)
        async with session_scope() as s:
            taken = (
                await s.execute(select(Workspace.user_id).where(Workspace.root_path == str(root)))
            ).scalar_one_or_none()
            if taken is not None:
                raise HTTPException(status.HTTP_409_CONFLICT, "that folder is already a workspace")
            ws = Workspace(id=workspace_id, user_id=user.id, name=body.name, root_path=str(root))
            s.add(ws)
        return WorkspaceOut.model_validate(ws)

    @r.get("/workspaces/{workspace_id}")
    async def get_workspace(user: auth.CurrentUser, workspace_id: uuid.UUID) -> WorkspaceOut:
        return WorkspaceOut.model_validate(await _workspace(user, workspace_id))

    @r.get("/workspaces/{workspace_id}/sessions")
    async def list_sessions(user: auth.CurrentUser, workspace_id: uuid.UUID) -> list[SessionOut]:
        ws = await _workspace(user, workspace_id)
        async with session_scope() as s:
            rows = await s.execute(
                select(Session).where(Session.repo_path == ws.root_path).order_by(Session.updated_at.desc()).limit(50)
            )
            return [SessionOut.model_validate(row) for row in rows.scalars()]

    @r.get("/workspaces/{workspace_id}/runs")
    async def list_runs(
        user: auth.CurrentUser, workspace_id: uuid.UUID, limit: Annotated[int, Query(ge=1, le=200)] = 20
    ) -> list[RunOut]:
        await _workspace(user, workspace_id)
        async with session_scope() as s:
            rows = await s.execute(
                select(Run).where(Run.workspace_id == workspace_id).order_by(Run.created_at.desc()).limit(limit)
            )
            return [RunOut.model_validate(run) for run in rows.scalars()]

    # ── runs ──

    @r.post("/workspaces/{workspace_id}/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(user: auth.CurrentUser, workspace_id: uuid.UUID, body: RunIn, pool: Pool) -> RunOut:
        ws = await _workspace(user, workspace_id)
        session_id = await _session_for(ws, body)
        worker = pool.workers.get(ws.id)
        if worker is not None and worker.busy:
            raise HTTPException(status.HTTP_409_CONFLICT, "a run is already in progress in this workspace")
        run_id = uuid.uuid4()
        async with session_scope() as s:
            run = Run(
                id=run_id, user_id=user.id, workspace_id=ws.id, session_id=uuid.UUID(session_id),
                kind=body.kind, input=body.input, status="running",
            )  # fmt: skip
            s.add(run)
        op = RunOp(run_id=str(run_id), kind=body.kind, input=body.input, session_id=session_id)
        try:
            await pool.start_run(user_id=user.id, workspace_id=ws.id, root=Path(ws.root_path), run_id=run_id, op=op)
        except PoolError as e:
            async with session_scope() as s:
                await s.execute(update(Run).where(Run.id == run_id).values(status="error", detail=e.detail))
            raise
        return RunOut.model_validate(await _run(user, run_id))

    @r.get("/runs/{run_id}")
    async def get_run(user: auth.CurrentUser, run_id: uuid.UUID) -> RunOut:
        return RunOut.model_validate(await _run(user, run_id))

    @r.get("/runs/{run_id}/events")
    async def run_events(
        user: auth.CurrentUser,
        run_id: uuid.UUID,
        pool: Pool,
        request: Request,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        await _run(user, run_id)
        if last_event_id is not None and last_event_id.isdigit():
            after = max(after, int(last_event_id))
        feed = pool.feeds.get(run_id)
        queue: asyncio.Queue[FeedItem | None]
        if feed is not None:
            backlog, queue = feed.subscribe(after)
        else:
            async with session_scope() as s:
                rows = await s.execute(
                    select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.seq > after).order_by(RunEvent.seq)
                )
                backlog = [FeedItem(e.seq, e.kind, e.data) for e in rows.scalars()]
            queue = asyncio.Queue()
            queue.put_nowait(None)

        async def stream() -> AsyncIterator[bytes]:
            try:
                for stored in backlog:
                    yield _sse(stored)
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
                    except TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield b": ping\n\n"
                        continue
                    if item is None:
                        return
                    yield _sse(item)
            finally:
                if feed is not None:
                    feed.unsubscribe(queue)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)

    @r.post("/runs/{run_id}/approvals/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def answer_approval(
        user: auth.CurrentUser, run_id: uuid.UUID, request_id: str, body: ApprovalIn, pool: Pool
    ) -> None:
        await _run(user, run_id)
        pool.approve(run_id, request_id, body.decisions)

    @r.post("/runs/{run_id}/plan-reviews/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def answer_plan_review(
        user: auth.CurrentUser, run_id: uuid.UUID, request_id: str, body: PlanReviewIn, pool: Pool
    ) -> None:
        await _run(user, run_id)
        op = PlanReviewOp(request_id=request_id, approved=body.approved, plan=body.plan, feedback=body.feedback)
        pool.answer(run_id, request_id, op)

    @r.post("/runs/{run_id}/stop", status_code=status.HTTP_202_ACCEPTED)
    async def stop_run(user: auth.CurrentUser, run_id: uuid.UUID, pool: Pool) -> dict[str, str]:
        await _run(user, run_id)
        pool.stop(run_id)
        return {"status": "stopping"}

    return r


def serve(host: str | None = None, port: int | None = None) -> int:
    """``kartrix serve``: run the API with Uvicorn."""
    import uvicorn

    host = host or settings.server.host
    port = port or settings.server.port
    if host not in ("127.0.0.1", "localhost", "::1") and settings.server.allow_local_paths:
        print(
            "kartrix serve: refusing to listen on a public address while server.allow_local_paths is true — "
            "anyone who signs up could point the agent at this machine's folders. Set it to false for hosting.",
        )
        return 3
    uvicorn.run(create_app(), host=host, port=port, log_level="info", proxy_headers=False)
    return 0
