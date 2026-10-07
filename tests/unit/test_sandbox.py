"""Sandbox (B8): shared policy, registry proxy, backend command lines, and — on Windows — a real
AppContainer run. Linux backends run for real in CI (and via Docker locally); here their command
lines and Landlock rules are checked without a Linux kernel."""

from __future__ import annotations

import asyncio
import base64
import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kartrix.config import settings
from kartrix.sandbox import manager, policy
from kartrix.sandbox.base import SandboxRun
from kartrix.sandbox.docker import DockerBackend
from kartrix.sandbox.posix import BubblewrapBackend, _allowed_tree, seatbelt_profile
from kartrix.sandbox.proxy import RegistryProxy, host_allowed
from kartrix.security import workspace as ws_mod
from kartrix.security.command_rules import Category
from kartrix.security.workspace import Workspace, set_workspace
from tests.fakes import FakeSandbox


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Workspace]:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n")
    (root / "src" / ".env.local").write_text("TOKEN=1\n")
    (root / ".env").write_text("SECRET=1\n")
    (root / ".env.example").write_text("SECRET=\n")
    (root / ".git" / "hooks").mkdir(parents=True)
    (root / ".kartrix" / "skills").mkdir(parents=True)
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / ".env").write_text("x")
    previous = ws_mod._current
    yield set_workspace(root)
    ws_mod._current = previous


def _run(ws: Workspace, network: str = "off", argv: list[str] | None = None) -> SandboxRun:
    argv = argv or [sys.executable, "-c", "print(1)"]
    return SandboxRun(argv, argv, ws.root, {"PATH": os.environ.get("PATH", "")}, network, ws.root)  # type: ignore[arg-type]


# ── shared policy ─────────────────────────────────────────────────────


def test_protected_paths(ws: Workspace) -> None:
    prot = policy.protected_paths(ws)
    rel = lambda paths: sorted(ws.relative(p) for p in paths)  # noqa: E731
    assert rel(prot.hidden) == [".env", ".git", "src/.env.local"]
    assert rel(prot.readonly) == [".kartrix", ".kartrix/skills"]  # readable skills, never writable
    assert ".env.example" not in rel(prot.hidden + prot.readonly)  # node_modules isn't walked either


def test_sandbox_env(tmp_path: Path) -> None:
    base = {"PATH": "/bin", "HTTPS_PROXY": "http://corp:3128"}
    off = policy.sandbox_env(base, tmp_path, "off")
    assert "HTTPS_PROXY" not in off and off["TMPDIR"] == str(tmp_path / "tmp")
    assert off["npm_config_cache"].startswith(str(tmp_path)) and off["UV_CACHE_DIR"].startswith(str(tmp_path))
    assert off["npm_config_userconfig"] == str(tmp_path / "home" / ".npmrc")  # never the user's tokens
    reg = policy.sandbox_env(base, tmp_path, "registries", "http://kartrix:t@127.0.0.1:1")
    assert reg["HTTPS_PROXY"] == reg["npm_config_https_proxy"] == "http://kartrix:t@127.0.0.1:1"
    assert "127.0.0.1" in reg["NO_PROXY"]
    assert policy.sandbox_env(base, tmp_path, "full")["HTTPS_PROXY"] == "http://corp:3128"


def test_home_secrets_exclude_workspace(ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "userhome"
    (home / ".ssh").mkdir(parents=True)
    (home / ".npmrc").write_text("//registry.npmjs.org/:_authToken=x")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    found = policy.home_secrets(ws.root)
    assert home / ".ssh" in found and home / ".npmrc" in found
    assert all(not str(p).startswith(str(ws.root) + os.sep) for p in found)


# ── manager ───────────────────────────────────────────────────────────


def test_network_for() -> None:
    enforced, loose = FakeSandbox(True), FakeSandbox(False)
    assert manager.network_for(Category.RUN, enforced) == "off"
    assert manager.network_for(Category.INSTALL, enforced) == "registries"
    assert manager.network_for(Category.INSTALL, loose) == "full"
    assert manager.network_for(Category.NETWORK, enforced) == "full"


def test_backend_none_and_docker_without_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.sandbox, "backend", "none")
    status = manager.detect()
    assert status.backend is None and "sandbox.backend: none" in status.describe()
    monkeypatch.setattr(settings.sandbox, "backend", "docker")
    status = manager.detect()
    assert status.backend is None and "sandbox.docker.image is not set" in status.describe()


# ── registry proxy ────────────────────────────────────────────────────


def test_host_allowed() -> None:
    allowed = ["pypi.org", "registry.npmjs.org"]
    assert host_allowed("pypi.org", allowed) and host_allowed("files.PYPI.org.", allowed)
    assert not host_allowed("evilpypi.org", allowed) and not host_allowed("pypi.org.evil.com", allowed)


@pytest.fixture
def proxy() -> Iterator[RegistryProxy]:
    p = RegistryProxy(["pypi.org"])
    p.start()
    yield p
    p.stop()


def _ask(proxy: RegistryProxy, request: str, then: bytes = b"") -> bytes:
    with socket.create_connection(("127.0.0.1", proxy.port), timeout=5) as s:
        s.sendall(request.encode())
        reply = s.recv(4096)
        if then and reply.startswith(b"HTTP/1.1 200"):
            s.sendall(then)
            reply += s.recv(4096)
        return reply


def _auth(token: str) -> str:
    return "Proxy-Authorization: Basic " + base64.b64encode(f"kartrix:{token}".encode()).decode() + "\r\n"


def test_proxy_requires_token(proxy: RegistryProxy) -> None:
    assert _ask(proxy, "CONNECT pypi.org:443 HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 407")
    assert _ask(proxy, f"CONNECT pypi.org:443 HTTP/1.1\r\n{_auth('wrong')}\r\n").startswith(b"HTTP/1.1 407")


@pytest.mark.parametrize(
    ("request_line", "status"),
    [("CONNECT evil.com:443", b"403"), ("CONNECT pypi.org:80", b"403"), ("CONNECT pypi.org.evil.com:443", b"403"),
     ("GET http://pypi.org/simple", b"405")],
)  # fmt: skip
def test_proxy_refuses(proxy: RegistryProxy, request_line: str, status: bytes) -> None:
    reply = _ask(proxy, f"{request_line} HTTP/1.1\r\n{_auth(proxy.token)}\r\n")
    assert reply.startswith(b"HTTP/1.1 " + status)


def test_proxy_tunnels_to_allowed_registry(proxy: RegistryProxy) -> None:
    seen: list[tuple[str, int]] = []

    async def fake_upstream(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        seen.append((host, port))
        server = await asyncio.start_server(_echo, "127.0.0.1", 0)
        return await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])

    async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"echo:" + await reader.read(100))
        await writer.drain()
        writer.close()

    proxy.open_upstream = fake_upstream  # type: ignore[assignment]
    reply = _ask(proxy, f"CONNECT pypi.org:443 HTTP/1.1\r\n{_auth(proxy.token)}\r\n", then=b"hello")
    assert reply.startswith(b"HTTP/1.1 200") and reply.endswith(b"echo:hello")
    assert seen == [("pypi.org", 443)]


# ── backend command lines ─────────────────────────────────────────────


def test_bubblewrap_args(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    launch = BubblewrapBackend(Path("/usr/bin/bwrap")).prepare(_run(ws))
    args = [str(a) for a in launch.args]
    assert args[0].replace("\\", "/") == "/usr/bin/bwrap" and "--unshare-net" in args and "--unshare-pid" in args
    joined = " ".join(args)
    assert f"--bind {ws.root} {ws.root}" in joined
    assert f"--ro-bind /dev/null {ws.root / '.env'}" in joined  # hidden file
    assert f"--tmpfs {ws.root / '.git'} --remount-ro {ws.root / '.git'}" in joined  # hidden folder
    assert f"--ro-bind {ws.root / '.kartrix'} {ws.root / '.kartrix'}" in joined  # read-only folder
    assert args[args.index("--") + 1] == sys.executable  # then the helper
    full = BubblewrapBackend(Path("/usr/bin/bwrap")).prepare(_run(ws, "full"))
    assert "--unshare-net" not in full.args


def test_seatbelt_profile_uses_parameters(tmp_path: Path) -> None:
    weird = tmp_path / 'a "quoted") (allow default'
    profile, params = seatbelt_profile([weird], [tmp_path / "ro"], [tmp_path / "secret"], "off")
    assert str(weird) not in profile and params["W0"] == os.path.realpath(weird)  # paths never in the text
    assert '(deny file-write* (subpath (param "R0")))' in profile
    assert '(deny file-read* file-write* (subpath (param "H0")))' in profile
    assert "(deny network*)" in profile and "com.apple.SecurityServer" in profile
    assert "(deny network*)" not in seatbelt_profile([weird], [], [], "full")[0]


def test_landlock_allowed_tree(tmp_path: Path) -> None:
    for d in ("a/b", "a/c", "d"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "a" / "secret").write_text("x")
    allowed = _allowed_tree(tmp_path, [tmp_path / "a" / "secret"])
    assert sorted(Path(p).relative_to(tmp_path).as_posix() for p in allowed) == ["a/b", "a/c", "d"]
    assert _allowed_tree(tmp_path, []) == [os.path.normpath(str(tmp_path))]


def test_docker_args(ws: Workspace) -> None:
    run = _run(ws, argv=["pytest", "-q"])
    args = DockerBackend("docker", "python:3.12-slim").prepare(run).args
    assert isinstance(args, list)
    joined = " ".join(args)
    assert "--network none" in joined and "--cap-drop ALL" in joined and "no-new-privileges" in joined
    assert f"{ws.root}:/workspace" in joined and f"{ws.root / '.kartrix'}:/workspace/.kartrix:ro" in joined
    assert "--tmpfs /workspace/.git:ro" in joined
    assert args[-3:] == ["python:3.12-slim", "pytest", "-q"]
    assert "--network bridge" in " ".join(DockerBackend("docker", "img").prepare(_run(ws, "full", ["npm", "ci"])).args)


# ── Windows: a real AppContainer ──────────────────────────────────────

_PROBE = r"""
import os, socket
from pathlib import Path
def t(name, fn):
    try:
        fn(); print(name, "ok")
    except OSError:
        print(name, "blocked")
t("write_ws", lambda: Path("out.txt").write_text("x"))
t("read_env", lambda: Path(".env").read_text())
t("write_git", lambda: Path(".git/hooks/pre-commit").write_text("evil"))
t("read_skill", lambda: os.listdir(".kartrix/skills"))
t("write_kartrix", lambda: Path(".kartrix/skills/x.md").write_text("x"))
t("write_outside", lambda: Path(os.environ["OUTSIDE"]).write_text("x"))
t("temp", lambda: Path(os.environ["TEMP"], "t.txt").write_text("x"))
t("net", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3).close())
"""


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_appcontainer_confines_a_real_process(ws: Workspace, tmp_path: Path) -> None:
    from kartrix.sandbox.windows import AppContainerBackend, container_name, delete_profile
    from kartrix.security.environment import scrubbed_env
    from kartrix.tools.process_runner import run_launch

    backend = AppContainerBackend.detect()
    argv = [sys.executable, "-c", _PROBE]
    env = {**scrubbed_env(), "OUTSIDE": str(tmp_path / "outside.txt")}
    try:
        result = run_launch(backend.prepare(SandboxRun(argv, argv, ws.root, env, "off", ws.root)), 60)
        assert result.returncode == 0, result.stderr
        assert dict(line.split() for line in result.stdout.splitlines()) == {
            "write_ws": "ok", "read_env": "blocked", "write_git": "blocked", "read_skill": "ok",
            "write_kartrix": "blocked", "write_outside": "blocked", "temp": "ok", "net": "blocked",
        }  # fmt: skip
    finally:
        backend.reset()
        delete_profile(container_name(ws.root))


# What each POSIX backend must block; Landlock can't protect the workspace's own files (see posix.py).
_POSIX_EXPECTED = {
    "bubblewrap": {"read_env": "blocked", "write_git": "blocked", "write_kartrix": "blocked"},
    "Seatbelt": {"read_env": "blocked", "write_git": "blocked", "write_kartrix": "blocked"},
    "Landlock": {"read_env": "ok", "write_git": "ok", "write_kartrix": "ok"},
}


def _posix_backend(name: str) -> object:
    from kartrix.sandbox import posix
    from kartrix.sandbox.base import SandboxUnavailable

    cls = {"bubblewrap": posix.BubblewrapBackend, "Landlock": posix.LandlockBackend, "Seatbelt": posix.SeatbeltBackend}[
        name
    ]
    try:
        return cls.detect()
    except SandboxUnavailable as e:
        pytest.skip(f"{name} not available: {e}")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sandboxes")
@pytest.mark.parametrize("name", ["bubblewrap", "Landlock", "Seatbelt"])
def test_posix_sandbox_confines_a_real_process(ws: Workspace, tmp_path: Path, name: str) -> None:
    from kartrix.security.environment import scrubbed_env
    from kartrix.tools.process_runner import run_launch

    backend = _posix_backend(name)
    argv = [sys.executable, "-c", _PROBE]
    env = {**scrubbed_env(), "OUTSIDE": str(tmp_path / "outside.txt")}
    launch = backend.prepare(SandboxRun(argv, argv, ws.root, env, "off", ws.root))  # type: ignore[attr-defined]
    result = run_launch(launch, 60)
    assert result.returncode == 0, result.stderr
    got = dict(line.split() for line in result.stdout.splitlines())
    got.pop("write_outside")  # bubblewrap: lands in the sandbox's private /tmp — check the host instead
    assert not (tmp_path / "outside.txt").exists()
    expected = {"write_ws": "ok", "read_skill": "ok", "temp": "ok", "net": "blocked", **_POSIX_EXPECTED[name]}
    assert got == expected
    if name == "Landlock":
        assert "changed protected files" in result.stderr  # the .git write is reported


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock is Linux-only")
def test_landlock_filesystem_rules_on_any_abi(ws: Workspace, tmp_path: Path) -> None:
    """The file rules work on every Landlock version (the backend itself also needs ABI 4 for the network)."""
    from kartrix.sandbox._helper import landlock_abi
    from kartrix.sandbox.posix import LandlockBackend
    from kartrix.security.environment import scrubbed_env
    from kartrix.tools.process_runner import run_launch

    if (abi := landlock_abi()) < 1:
        pytest.skip("Landlock is not enabled in this kernel")
    argv = [sys.executable, "-c", _PROBE]
    env = {**scrubbed_env(), "OUTSIDE": str(tmp_path / "outside.txt")}
    result = run_launch(LandlockBackend(abi).prepare(SandboxRun(argv, argv, ws.root, env, "full", ws.root)), 60)
    got = dict(line.split() for line in result.stdout.splitlines())
    assert (got["write_ws"], got["temp"], got["write_outside"]) == ("ok", "ok", "blocked"), result.stderr
    assert not (tmp_path / "outside.txt").exists()


# ── CLI and self-check ────────────────────────────────────────────────


def test_cli_sandbox_status(capsys: pytest.CaptureFixture[str]) -> None:
    from kartrix.cli import main

    assert main(["sandbox"]) == 0
    assert "Sandbox: fake" in capsys.readouterr().out
    manager.set_sandbox(None)
    assert main(["sandbox", "status"]) == 1
    assert "needs your approval" in capsys.readouterr().out


def test_self_check_flags_a_sandbox_that_confines_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartrix.cli import main
    from kartrix.sandbox.selfcheck import run_check

    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    results = {r.name: r for r in run_check(FakeSandbox())}  # pass-through: everything is allowed
    assert results["write_workspace"].ok and results["write_temp"].ok
    assert not results["read_env"].ok and not results["write_git"].ok and not results["write_outside"].ok
    assert main(["sandbox", "check"]) == 1
    assert "Sandbox check FAILED: read_env, write_git, write_outside" in capsys.readouterr().out
