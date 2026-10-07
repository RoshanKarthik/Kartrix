"""Local HTTPS proxy that only reaches the allow-listed package registries (B8).

Sandboxed installs on macOS and Linux (bubblewrap) have no network except this proxy:

- only ``CONNECT host:443`` (TLS end to end — the proxy never sees the traffic), and only to a
  host in ``permissions.registries`` or one of its subdomains;
- every request must carry the proxy's random token (``Proxy-Authorization``), which is put in
  the environment of ``registries`` runs only — commands with network "off" can reach the port
  (loopback) but not use it;
- it runs on a background thread for the life of the Kartrix process, listening on 127.0.0.1 and,
  for network-namespaced sandboxes, on a Unix socket bound into the sandbox.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import secrets
import sys
import threading
from pathlib import Path

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_HEAD_LIMIT = 16 * 1024
_HEAD_TIMEOUT = 30
_CONNECT_TIMEOUT = 15
_USER = "kartrix"


def host_allowed(host: str, allowed: list[str]) -> bool:
    host = host.lower().strip().rstrip(".")
    return any(host == a.lower() or host.endswith("." + a.lower()) for a in allowed)


class RegistryProxy:
    def __init__(self, allowed_hosts: list[str]) -> None:
        self.allowed = list(allowed_hosts)
        self.token = secrets.token_urlsafe(24)
        self.port = 0
        self.unix_path: Path | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._servers: list[asyncio.Server] = []
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.open_upstream = asyncio.open_connection  # replaced in tests

    @property
    def url(self) -> str:
        return f"http://{_USER}:{self.token}@127.0.0.1:{self.port}"

    def start(self, unix_path: Path | None = None) -> None:
        thread = threading.Thread(target=self._run, args=(unix_path,), name="kartrix-registry-proxy", daemon=True)
        thread.start()
        self._ready.wait(10)
        if self._error is not None:
            raise RuntimeError(f"registry proxy failed to start: {self._error}") from self._error
        if not self._ready.is_set():
            raise RuntimeError("registry proxy failed to start in time")

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._close_servers(), loop).result(5)
            except Exception as e:  # stopping anyway
                logger.debug("Registry proxy didn't close cleanly", extra={"error": str(e)})
            loop.call_soon_threadsafe(loop.stop)
        if self.unix_path is not None:
            self.unix_path.unlink(missing_ok=True)

    async def _close_servers(self) -> None:
        for server in self._servers:
            server.close()
        await asyncio.sleep(0.05)  # let the cancelled accept tasks finish

    def _run(self, unix_path: Path | None) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)  # this thread's loop
        self._loop = loop
        try:
            server = loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
            self._servers.append(server)
            self.port = server.sockets[0].getsockname()[1]
            if unix_path is not None and sys.platform != "win32":
                unix_path.unlink(missing_ok=True)
                self._servers.append(loop.run_until_complete(asyncio.start_unix_server(self._handle, str(unix_path))))
                unix_path.chmod(0o600)
                self.unix_path = unix_path
        except BaseException as e:  # reported to start()
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def _authorized(self, value: str) -> bool:
        expected = "Basic " + base64.b64encode(f"{_USER}:{self.token}".encode()).decode()
        return hmac.compare_digest(value.strip().encode(), expected.encode())

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), _HEAD_TIMEOUT)
        except (TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        if len(head) > _HEAD_LIMIT:
            await _reply(writer, 431, "Request Header Fields Too Large")
            return
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        headers = {k.strip().lower(): v for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
        if not self._authorized(headers.get("proxy-authorization", "")):
            await _reply(writer, 407, "Proxy Authentication Required", 'Proxy-Authenticate: Basic realm="kartrix"\r\n')
            return
        if len(parts) != 3 or parts[0] != "CONNECT":
            await _reply(writer, 405, "Method Not Allowed (only HTTPS to package registries)")
            return
        host, _, port = parts[1].rpartition(":")
        host = host.strip("[]")
        if port != "443" or not host_allowed(host, self.allowed):
            logger.warning("Sandbox proxy refused a connection", extra={"target": parts[1][:200]})
            await _reply(writer, 403, f"Forbidden: {parts[1][:200]} is not an allowed package registry")
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(self.open_upstream(host, 443), _CONNECT_TIMEOUT)
        except (OSError, TimeoutError) as e:
            await _reply(writer, 502, f"Bad Gateway ({type(e).__name__})")
            return
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def _reply(writer: asyncio.StreamWriter, status: int, text: str, extra: str = "") -> None:
    try:
        writer.write(f"HTTP/1.1 {status} {text}\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except (ConnectionError, OSError):
            pass


_proxy: RegistryProxy | None = None
_lock = threading.Lock()


def get_proxy() -> RegistryProxy:
    """The process-wide proxy, started on first use with ``permissions.registries``
    (plus a Unix socket on Linux, for sandboxes with their own network namespace)."""
    global _proxy
    with _lock:
        if _proxy is None:
            import os

            from kartrix.config import settings
            from kartrix.paths import user_cache_dir

            unix_path = None
            if sys.platform.startswith("linux"):
                (user_cache_dir() / "sandbox").mkdir(parents=True, exist_ok=True)
                unix_path = user_cache_dir() / "sandbox" / f"proxy-{os.getpid()}.sock"
            proxy = RegistryProxy(settings.permissions.registries)
            proxy.start(unix_path)
            _proxy = proxy
            logger.info("Registry proxy started", extra={"port": proxy.port, "hosts": proxy.allowed})
        return _proxy
