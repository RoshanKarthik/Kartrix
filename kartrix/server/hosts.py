"""The ``Host`` check of a server on a loopback address.

A web page can point a domain of its own at 127.0.0.1 (DNS rebinding) and then call the local API as a same-origin
page: sign up, register a folder of this machine as a workspace and run the agent on it. The browser still sends that
domain in ``Host``, so a loopback server answers only loopback names (and ``*.localhost``, which browsers resolve to
loopback themselves). A server on a public address has no such check: it sits behind its platform's proxy and domain.
"""

from __future__ import annotations

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def host_name(header: str) -> str:
    """The name in a ``Host`` header, without the port: ``127.0.0.1:8765`` → ``127.0.0.1``, ``[::1]:80`` → ``::1``."""
    header = header.strip().lower()
    if header.startswith("["):
        end = header.find("]")
        return header[1:end] if end != -1 else header
    name, colon, port = header.rpartition(":")
    return name if colon and port.isdigit() and ":" not in name else header


def is_loopback_name(name: str) -> bool:
    return name in LOOPBACK or name.endswith(".localhost")


class LoopbackHostsOnly:
    """ASGI middleware: a request whose ``Host`` isn't a loopback name gets 400 and never reaches a route."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            headers: dict[bytes, bytes] = dict(scope.get("headers") or [])
            if not is_loopback_name(host_name(headers.get(b"host", b"").decode("latin-1"))):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                response = PlainTextResponse(
                    "invalid Host header — this Kartrix server answers on 127.0.0.1 and localhost only", 400
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
