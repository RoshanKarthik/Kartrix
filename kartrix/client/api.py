"""The API client used by the CLI (and a reference for the web app and VS Code extension).

Credentials (server URL + token) live in ``<user data dir>/credentials.json`` with owner-only
permissions; ``KARTRIX_SERVER_URL`` / ``KARTRIX_API_TOKEN`` override them (CI, scripts) — not ``KARTRIX_SERVER``,
which the config reads as the whole ``server`` section.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from kartrix.paths import user_data_dir

RECONNECTS = 5  # an event stream that drops is resumed from the last event this many times


class ApiError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{detail} (HTTP {status})")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Credentials:
    server: str
    token: str
    email: str = ""


def credentials_file() -> Path:
    return user_data_dir() / "credentials.json"


def load_credentials() -> Credentials | None:
    server, token = os.environ.get("KARTRIX_SERVER_URL"), os.environ.get("KARTRIX_API_TOKEN")
    if server and token:
        return Credentials(server.rstrip("/"), token)
    path = credentials_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Credentials(str(data["server"]).rstrip("/"), str(data["token"]), str(data.get("email", "")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_credentials(creds: Credentials) -> Path:
    path = credentials_file()
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"server": creds.server, "token": creds.token, "email": creds.email}, f)
    os.replace(tmp, path)
    return path


def forget_credentials() -> bool:
    try:
        credentials_file().unlink()
    except FileNotFoundError:
        return False
    return True


@dataclass
class SseMessage:
    kind: str
    data: dict[str, Any]
    seq: int | None


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or response.reason_phrase or "request failed")


class KartrixClient:
    def __init__(self, server: str, token: str | None = None, *, transport: httpx.AsyncBaseTransport | None = None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.server = server.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self.server + "/v1", headers=headers, timeout=httpx.Timeout(30.0), transport=transport
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> KartrixClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _call(self, method: str, path: str, json_body: Any = None, **params: Any) -> Any:
        response = await self._http.request(method, path, json=json_body, params=params or None)
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))
        return response.json() if response.content else None

    # accounts
    async def signup(self, email: str, password: str, client_name: str = "cli") -> dict[str, Any]:
        body = {"email": email, "password": password, "token_name": client_name}
        result: dict[str, Any] = await self._call("POST", "/auth/signup", body)
        return result

    async def login(self, email: str, password: str, client_name: str = "cli") -> dict[str, Any]:
        body = {"email": email, "password": password, "token_name": client_name}
        result: dict[str, Any] = await self._call("POST", "/auth/login", body)
        return result

    async def logout(self) -> None:
        await self._call("POST", "/auth/logout")

    async def me(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("GET", "/me")
        return result

    async def models(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("GET", "/models")
        return result

    # workspaces
    async def workspaces(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = await self._call("GET", "/workspaces")
        return result

    async def create_workspace(self, name: str, path: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("POST", "/workspaces", {"name": name, "path": path})
        return result

    async def command(self, workspace_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A workspace command (kartrix.core.commands) carried out by the server; its result."""
        # the server may start the workspace's agent first (up to 180 s), then 60 s per command (300 s for /connect)
        timeout = httpx.Timeout(540.0 if name == "mcp.connect" else 300.0)
        response = await self._http.post(
            f"/workspaces/{workspace_id}/commands", json={"name": name, "args": args}, timeout=timeout
        )
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))
        result: dict[str, Any] = response.json()["result"]
        return result

    # runs
    async def start_run(
        self,
        workspace_id: str,
        kind: str,
        text: str,
        *,
        session_id: str | None = None,
        new_session: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        body = {"kind": kind, "input": text, "session_id": session_id, "new_session": new_session, "model": model}
        result: dict[str, Any] = await self._call("POST", f"/workspaces/{workspace_id}/runs", body)
        return result

    async def run(self, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await self._call("GET", f"/runs/{run_id}")
        return result

    async def answer_approval(self, run_id: str, request_id: str, decisions: list[dict[str, Any]]) -> None:
        await self._call("POST", f"/runs/{run_id}/approvals/{request_id}", {"decisions": decisions})

    async def answer_plan_review(
        self, run_id: str, request_id: str, approved: bool, plan: dict[str, Any] | None = None, feedback: str = ""
    ) -> None:
        body = {"approved": approved, "plan": plan, "feedback": feedback}
        await self._call("POST", f"/runs/{run_id}/plan-reviews/{request_id}", body)

    async def stop(self, run_id: str) -> None:
        await self._call("POST", f"/runs/{run_id}/stop")

    async def events(self, run_id: str) -> AsyncIterator[SseMessage]:
        """The run's feed from the start until it ends; a dropped connection resumes after the last event."""
        last = 0
        failures = 0
        while True:
            try:
                headers = {"Last-Event-ID": str(last)} if last else {}
                timeout = httpx.Timeout(30.0, read=None)
                async with self._http.stream("GET", f"/runs/{run_id}/events", headers=headers, timeout=timeout) as r:
                    if r.status_code >= 400:
                        await r.aread()
                        raise ApiError(r.status_code, _detail(r))
                    async for message in _parse_sse(r.aiter_lines()):
                        failures = 0
                        if message.seq is not None:
                            last = message.seq
                        yield message
                        if message.kind == "run_finished":
                            return
                return  # the server closed a finished run's feed
            except (httpx.TransportError, httpx.StreamError):
                failures += 1
                if failures > RECONNECTS:
                    raise


async def _parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[SseMessage]:
    kind, seq = "message", None
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield SseMessage(kind, json.loads("\n".join(data)), seq)
            kind, data, seq = "message", [], None
            continue
        if line.startswith(":"):
            continue  # heartbeat
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            kind = value
        elif name == "data":
            data.append(value)
        elif name == "id" and value.isdigit():
            seq = int(value)
