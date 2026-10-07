"""MCP hardening (B7): pinned config, verified binaries, tool vetting, persistent opt-in sessions."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kartrix.mcp import binaries
from kartrix.mcp.binaries import Asset, BinaryError, BinarySpec, ensure_binary, executable_name, platform_key
from kartrix.mcp.mcp_client import McpError, McpManager, enabled_servers
from kartrix.mcp.mcp_config import McpConfigError, McpServer, command_is_pinned, load_mcp_configs
from kartrix.security import external_tools

FAKE_SERVER = str(Path(__file__).resolve().parent.parent / "fake_mcp_server.py")


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    external_tools.clear()
    yield tmp_path / "home"
    external_tools.clear()


def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, servers: dict[str, Any]) -> Path:
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({"mcp_servers": servers}), encoding="utf-8")
    monkeypatch.setattr("kartrix.mcp.mcp_config._CONFIG_PATH", path)
    return path


# ── config ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "args", "pinned"),
    [
        ("npx", ["-y", "@scope/server@1.2.3"], True),
        ("npx", ["-y", "server@1.2.3-beta.1", "--flag"], True),
        ("npx", ["-y", "@scope/server"], False),
        ("npx", ["-y", "server@latest"], False),
        ("npx", ["-y", "server@^1.2.0"], False),
        ("npx.cmd", ["-p", "server@2.0.0", "server-bin"], True),
        ("uvx", ["mcp-server-git==0.6.2"], True),
        ("uvx", ["--from", "mcp-server-git==0.6.2", "mcp-server-git"], True),
        ("uvx", ["mcp-server-git"], False),
        ("uvx", ["mcp-server-git>=0.6"], False),
        ("node", ["server.js"], False),
        ("docker", ["run", "-i", "image:latest"], False),
    ],
)
def test_pinning_rules(command: str, args: list[str], pinned: bool) -> None:
    assert command_is_pinned(command, args) is pinned


def test_config_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(tmp_path, monkeypatch, {"a": {"command": "node", "args": ["s.js"]}})
    with pytest.raises(McpConfigError, match="not pinned"):
        load_mcp_configs()
    write_config(tmp_path, monkeypatch, {"a": {"command": "node", "args": ["s.js"], "allow_unpinned": True}})
    assert load_mcp_configs()["a"].allow_unpinned

    for bad in (
        {"command": "uvx", "args": ["x==1.0.0"], "tool": {}},  # typo'd key
        {"command": "uvx", "args": ["x==1.0.0"], "binary": {}},  # both
        {"binary": {"name": "x", "version": "1.0.0", "url": "http://insecure/{file}", "executable": "x",
                    "assets": {"linux-x86_64": {"file": "x.tar.gz", "sha256": "0" * 64}}}},
        {"binary": {"name": "x", "version": "1.0.0", "url": "https://h/{file}", "executable": "x",
                    "assets": {"linux-x86_64": {"file": "x.tar.gz", "sha256": "not-a-hash"}}}},
    ):  # fmt: skip
        write_config(tmp_path, monkeypatch, {"a": bad})
        with pytest.raises(McpConfigError):
            load_mcp_configs()
    write_config(tmp_path, monkeypatch, {"Bad Name": {"command": "uvx", "args": ["x==1.0.0"]}})
    with pytest.raises(McpConfigError, match="server names"):
        load_mcp_configs()


def test_env_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    server = McpServer(command="uvx", args=["x==1.0.0"], env={"TOKEN": "${MY_TOKEN}", "DIR": "${MY_DIR}", "K": "v"})
    monkeypatch.delenv("MY_TOKEN", raising=False)
    monkeypatch.setenv("MY_DIR", 'D:\\proj\\"quoted"')
    assert server.resolved_env() == {"DIR": 'D:\\proj\\"quoted"', "K": "v"}  # unset → left out


def test_shipped_config_pins_the_official_github_binary() -> None:
    github = load_mcp_configs()["github"]
    assert github.binary is not None and github.command is None
    assert github.binary.url.startswith("https://github.com/github/github-mcp-server/")
    assert {"windows-x86_64", "darwin-arm64", "linux-x86_64"} <= set(github.binary.assets)
    assert "--read-only" in github.args and "--lockdown-mode" in github.args


# ── verified binaries ─────────────────────────────────────────────────


def _spec(archive: bytes, file: str) -> BinarySpec:
    asset = Asset(file, hashlib.sha256(archive).hexdigest())
    return BinarySpec("tool", "1.0.0", "https://example.invalid/{version}/{file}", "tool", {platform_key(): asset})


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tgz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize("kind", ["zip", "tgz"])
def test_binary_is_verified_and_only_the_executable_extracted(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = "tool.exe" if sys.platform == "win32" else "tool"
    members = {f"dist/{exe}": b"BINARY", "../../evil.txt": b"x", "README.md": b"doc"}
    archive = _zip(members) if kind == "zip" else _tgz(members)
    spec = _spec(archive, f"tool.{'zip' if kind == 'zip' else 'tar.gz'}")
    calls: list[str] = []
    monkeypatch.setattr(binaries, "_download", lambda url: calls.append(url) or archive)

    path = ensure_binary(spec)
    assert path.read_bytes() == b"BINARY" and path.name == executable_name(spec)
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # nothing else written
    assert calls == [f"https://example.invalid/1.0.0/{spec.assets[platform_key()].file}"]
    ensure_binary(spec)
    assert len(calls) == 1  # cached


def test_tampered_download_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    good = _zip({"tool": b"BINARY", "tool.exe": b"BINARY"})
    spec = _spec(good, "tool.zip")
    monkeypatch.setattr(binaries, "_download", lambda url: _zip({"tool": b"EVIL", "tool.exe": b"EVIL"}))
    with pytest.raises(BinaryError, match="failed verification"):
        ensure_binary(spec)
    assert not binaries.install_dir(spec).exists() or not any(binaries.install_dir(spec).iterdir())


def test_unsupported_platform_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = BinarySpec("tool", "1.0.0", "https://x/{file}", "tool", {"plan9-mips": Asset("t.zip", "0" * 64)})
    with pytest.raises(BinaryError, match="no build for this platform"):
        ensure_binary(spec)

    attempts = []

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *a: object) -> None:
            pass

    def flaky(url: str, timeout: float) -> Response:
        attempts.append(url)
        if len(attempts) < 3:
            raise ConnectionResetError("reset by peer")
        return Response(b"data")

    monkeypatch.setattr(binaries.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(binaries.time, "sleep", lambda s: None)
    assert binaries._download("https://x/a") == b"data" and len(attempts) == 3


# ── sessions and vetting (real stdio server) ──────────────────────────


@pytest.fixture
def fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = {
        "command": sys.executable,
        "args": [FAKE_SERVER],
        "allow_unpinned": True,
        "tools": {"allow": ["server_pid", "get_issue", "create_issue", "read_file", "helpful"], "approve": []},
    }
    write_config(tmp_path, monkeypatch, {"fake": server})


@pytest.mark.usefixtures("fake_config")
async def test_session_vetting_and_opt_in() -> None:
    mcp = McpManager(["read_file", "run_command"])
    with pytest.raises(McpError, match="unknown MCP server"):
        await mcp.connect("nope")
    assert enabled_servers() == []

    tools = {t.name: t for t in await mcp.connect("fake")}
    try:
        # read_file would shadow the native tool; helpful's description is a prompt injection
        assert set(tools) == {"server_pid", "get_issue", "create_issue"}
        create, issue = external_tools.get("create_issue"), external_tools.get("get_issue")
        assert create is not None and create.needs_approval and "fake" in create.reason
        assert issue is not None and not issue.needs_approval  # readOnlyHint
        assert enabled_servers() == ["fake"]  # remembered for the next start

        pids = [(await tools["server_pid"].ainvoke({}))[0]["text"] for _ in range(2)]
        assert pids[0] == pids[1]  # one persistent server process, not one per call
    finally:
        await mcp.disconnect("fake")
    assert external_tools.get("get_issue") is None and enabled_servers() == []


async def test_allow_list_and_forced_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = {
        "command": sys.executable,
        "args": [FAKE_SERVER],
        "allow_unpinned": True,
        "tools": {"allow": ["get_issue"], "approve": ["get_issue"]},
    }
    write_config(tmp_path, monkeypatch, {"fake": server})
    mcp = McpManager([])
    try:
        assert [t.name for t in await mcp.connect("fake", remember=False)] == ["get_issue"]
        ext = external_tools.get("get_issue")
        assert ext is not None and ext.needs_approval
    finally:
        await mcp.close()
    assert enabled_servers() == []


async def test_failed_start_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = {"command": sys.executable, "args": ["-c", "import sys; sys.exit(3)"], "allow_unpinned": True}
    write_config(tmp_path, monkeypatch, {"broken": server})
    mcp = McpManager([])
    with pytest.raises(McpError, match="could not start the broken MCP server"):
        await mcp.connect("broken")
    assert mcp.connected == [] and enabled_servers() == []
