"""The web UI's files as ``kartrix serve`` hands them out, and the Host check of a loopback server. The pages' own
logic (feed parsing, input parsing, resuming a feed) is tested with Node in ``tests/web/``, run from here."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from kartrix import web
from kartrix.config import settings
from kartrix.server import app as server_app
from kartrix.server.app import create_app
from kartrix.server.hosts import LoopbackHostsOnly, host_name, is_loopback_name

REPO = Path(__file__).resolve().parents[2]


def _client(*, loopback_only: bool = False, host: str = "127.0.0.1:8765") -> httpx.AsyncClient:
    # No lifespan: serving files needs neither the database nor workers.
    transport = httpx.ASGITransport(app=create_app(loopback_only=loopback_only))
    return httpx.AsyncClient(transport=transport, base_url=f"http://{host}")


async def test_the_page_is_served_with_a_strict_policy() -> None:
    async with _client() as http:
        r = await http.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "frame-ancestors 'none'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert '<script type="module" src="/assets/app.js">' in r.text


async def test_every_file_the_page_loads_is_served() -> None:
    page = (web.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    scripts = "\n".join(p.read_text(encoding="utf-8") for p in web.STATIC_DIR.glob("*.js"))
    referenced = set(re.findall(r"/assets/([\w.-]+)", page + scripts))
    referenced |= {name + ".js" for name in re.findall(r'from "\./([\w-]+)\.js"', scripts)}
    assert {"app.js", "app.css", "api.js", "feed.js", "kartrix.svg"} <= referenced
    types = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
    async with _client() as http:
        for name in sorted(referenced):
            r = await http.get(f"/assets/{name}")
            assert r.status_code == 200, name
            assert r.headers["content-type"].startswith(types[Path(name).suffix]), name
            assert r.headers["x-content-type-options"] == "nosniff"
        assert (await http.get("/favicon.ico")).headers["content-type"] == "image/svg+xml"


@pytest.mark.parametrize(
    "path", ["/assets/nope.js", "/assets/index.html", "/assets/..%2F__init__.py", "/assets/%2E%2E%2Fapp.py"]
)
async def test_only_the_listed_files_are_served(path: str) -> None:
    async with _client() as http:
        assert (await http.get(path)).status_code == 404


def test_the_ui_loads_nothing_from_elsewhere_and_never_renders_html() -> None:
    for path in web.STATIC_DIR.iterdir():
        if path.suffix in (".html", ".js", ".css"):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"https?://", text), f"{path.name} refers to another origin"
            assert not re.search(
                r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function", text
            ), path.name
            assert "<script>" not in text and "style=" not in text, f"{path.name}: inline code the CSP would block"


def test_the_token_never_goes_in_a_url() -> None:
    scripts = "\n".join(p.read_text(encoding="utf-8") for p in web.STATIC_DIR.glob("*.js"))
    assert "new EventSource" not in scripts  # it can't send headers; the token would have to go in the URL
    assert "document.cookie" not in scripts
    assert not re.search(r"[?&]token=", scripts)


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("127.0.0.1:8765", "127.0.0.1"),
        ("LOCALHOST", "localhost"),
        ("[::1]:8765", "::1"),
        ("[::1]", "::1"),
        ("kartrix.localhost:80", "kartrix.localhost"),
        ("evil.example:8765", "evil.example"),
        ("", ""),
    ],
)
def test_host_name(header: str, name: str) -> None:
    assert host_name(header) == name


def test_loopback_names() -> None:
    assert all(is_loopback_name(n) for n in ("127.0.0.1", "localhost", "::1", "app.localhost"))
    assert not any(is_loopback_name(n) for n in ("", "evil.example", "localhost.evil.example", "127.0.0.2", "0.0.0.0"))  # noqa: S104


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"])
async def test_a_loopback_server_answers_loopback_names(host: str) -> None:
    async with _client(loopback_only=True, host=host) as http:
        assert (await http.get("/")).status_code == 200
        assert (await http.get("/v1/me")).status_code == 401  # reached the API (no token)


@pytest.mark.parametrize("host", ["rebound.example:8765", "127.0.0.1.nip.io:8765"])
async def test_a_loopback_server_refuses_other_names(host: str) -> None:
    """DNS rebinding: a page whose domain now resolves to 127.0.0.1 can't reach the API or the UI."""
    async with _client(loopback_only=True, host=host) as http:
        for method, path in [("GET", "/"), ("POST", "/v1/auth/signup"), ("GET", "/v1/health")]:
            r = await http.request(method, path, json={} if method == "POST" else None)
            assert r.status_code == 400, path
            assert "invalid Host header" in r.text


async def test_without_the_check_any_host_is_answered() -> None:
    async with _client(host="kartrix.example.com") as http:
        assert (await http.get("/")).status_code == 200


def _serve(monkeypatch: pytest.MonkeyPatch, host: str) -> tuple[int, dict[str, Any]]:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    monkeypatch.setattr(server_app, "harden_for_hosting", lambda: None)
    return server_app.serve(host=host, port=8799), seen


def test_serve_on_loopback_checks_hosts_and_prints_the_ui_address(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(settings.server, "allow_local_paths", True)
    code, seen = _serve(monkeypatch, "127.0.0.1")
    assert code == 0
    assert LoopbackHostsOnly in [m.cls for m in seen["app"].user_middleware]
    assert "web UI at http://127.0.0.1:8799/" in capsys.readouterr().out


def test_a_hosted_server_answers_its_own_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.server, "allow_local_paths", False)
    code, seen = _serve(monkeypatch, "0.0.0.0")  # noqa: S104 — a hosted server behind its platform's proxy
    assert code == 0
    assert LoopbackHostsOnly not in [m.cls for m in seen["app"].user_middleware]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_pages_logic_with_node() -> None:
    node = shutil.which("node")
    assert node is not None
    result = subprocess.run(
        [node, "--test", *map(str, sorted((REPO / "tests" / "web").glob("*.test.mjs")))],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
