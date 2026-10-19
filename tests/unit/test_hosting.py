"""Running as a hosted server: database URLs from hosting providers, hosted hardening, the CLI's admin command."""

from __future__ import annotations

import pytest

from kartrix.config import settings
from kartrix.db.engine import DatabaseConfigError, normalize_database_url


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgres://u:p@db.example:5432/k", "postgresql+asyncpg://u:p@db.example:5432/k"),
        ("postgresql://u:p@h/k?sslmode=require", "postgresql+asyncpg://u:p@h/k?ssl=require"),
        ("postgresql://u:p@h/k?sslmode=require&channel_binding=require", "postgresql+asyncpg://u:p@h/k?ssl=require"),
        ("postgresql+asyncpg://u:p@127.0.0.1:5432/k", "postgresql+asyncpg://u:p@127.0.0.1:5432/k"),
    ],
)
def test_provider_database_urls(given: str, expected: str) -> None:
    assert normalize_database_url(given) == expected


def test_non_postgres_urls_are_refused() -> None:
    with pytest.raises(DatabaseConfigError):
        normalize_database_url("mysql://u:p@h/k")


def test_hosted_servers_refuse_code_they_cannot_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from kartrix.server.app import harden_for_hosting

    monkeypatch.setattr(settings.sandbox, "backend", "auto")
    monkeypatch.delenv("KARTRIX_SANDBOX__BACKEND", raising=False)
    harden_for_hosting()
    assert settings.sandbox.backend == "native"  # no sandbox → refused, never offered for approval
    assert os.environ["KARTRIX_SANDBOX__BACKEND"] == "native"  # workers inherit it
    monkeypatch.setattr(settings.sandbox, "backend", "docker")
    harden_for_hosting()
    assert settings.sandbox.backend == "docker"  # an explicit sandbox choice is kept


def test_unsandboxable_commands_are_denied_under_native(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from pathlib import Path

    from kartrix.sandbox.manager import reset_sandbox, set_sandbox
    from kartrix.security import command_policy
    from kartrix.security import workspace as ws_mod

    root = Path(str(tmp_path))
    ws_mod.set_workspace(root)
    monkeypatch.setattr(settings.sandbox, "backend", "native")
    set_sandbox(None)
    try:
        decision = command_policy.evaluate("python hello.py", log=False)
    finally:
        reset_sandbox()
        ws_mod._current = None
    assert decision.action == "deny" and "native" in decision.reason


def test_port_from_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.server import app as server_app

    seen: dict[str, object] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr(settings.server, "allow_local_paths", True)
    monkeypatch.setenv("PORT", "10000")
    assert server_app.serve() == 0
    assert seen["port"] == 10000 and seen["host"] == "127.0.0.1"
    monkeypatch.setattr(settings.server, "allow_local_paths", True)
    assert server_app.serve(host="0.0.0.0") == 3  # noqa: S104 — public + local folders: refused


def test_cli_credentials_from_env_do_not_clash_with_config(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from kartrix.client.api import load_credentials
    from kartrix.config import load_settings

    monkeypatch.setenv("KARTRIX_SERVER_URL", "https://kartrix.example/")
    monkeypatch.setenv("KARTRIX_API_TOKEN", "kx_test")
    load_settings()  # KARTRIX_SERVER would be read as the whole `server` section and fail here
    creds = load_credentials()
    assert creds is not None and creds.server == "https://kartrix.example" and creds.token == "kx_test"
