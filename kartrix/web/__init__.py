"""The web UI that ``kartrix serve`` hands out at ``/``: sign-in, workspaces and chat with a run's live feed.

It is a thin client over the same HTTP API and SSE feed as the CLI and the VS Code extension — plain HTML, CSS and
JavaScript modules in ``static/`` (no build step, nothing fetched from elsewhere). It signs in with
``POST /v1/auth/login`` like the CLI and sends the token as a bearer header, keeping it in the browser's per-origin
storage rather than a cookie: browsers send a cookie for 127.0.0.1 to every port of it — including the apps Kartrix
builds and runs there — and a cookie would be an ambient credential for other sites to forge requests with.
Everything from the agent is rendered as text, and a strict Content-Security-Policy runs only these files.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)

HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",  # revalidate, so an updated Kartrix serves its new files at once
}

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}


@cache
def assets() -> frozenset[str]:
    """The files under ``/assets/``: a fixed list, so no request path ever reaches the file system."""
    return frozenset(p.name for p in STATIC_DIR.iterdir() if p.is_file() and p.suffix in _TYPES)


def _file(name: str) -> FileResponse:
    path = STATIC_DIR / name
    return FileResponse(path, media_type=_TYPES[path.suffix], headers=HEADERS)


def router() -> APIRouter:
    r = APIRouter(include_in_schema=False)

    @r.get("/")
    async def index() -> FileResponse:
        return _file("index.html")

    @r.get("/assets/{name}")
    async def asset(name: str) -> FileResponse:
        if name not in assets() or name == "index.html":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        return _file(name)

    @r.get("/favicon.ico")
    async def favicon() -> FileResponse:
        return _file("kartrix.svg")

    return r
