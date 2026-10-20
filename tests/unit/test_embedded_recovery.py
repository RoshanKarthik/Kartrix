"""The built-in database recovers from a crashed or killed Kartrix process (kartrix.db.embedded): processes that are
gone are dropped from the server's user list, and a server that runs but drops every connection is restarted once."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kartrix.db import embedded


def test_users_that_no_longer_run_are_forgotten(tmp_path: Path) -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    users = tmp_path / ".handle_pids.json"
    users.write_text(json.dumps([gone.pid, os.getpid()]))
    embedded._forget_dead_users(tmp_path)
    assert json.loads(users.read_text()) == [os.getpid()]
    embedded._forget_dead_users(tmp_path / "missing")  # no list yet: nothing to do
    users.write_text("not json")
    embedded._forget_dead_users(tmp_path)  # left alone
    assert users.read_text() == "not json"


def test_a_slow_recovery_is_waited_for_then_joined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import embedded_postgres
    from embedded_postgres import utils
    from embedded_postgres.postgres_server import PostgresServer

    calls: list[Path] = []
    joined = object()

    def get_server(pgdata: Path, cleanup_mode: str) -> Any:
        calls.append(pgdata)
        if len(calls) == 1:
            PostgresServer._instances[pgdata.resolve()] = "half-made"  # what the library leaves behind
            raise subprocess.TimeoutExpired("pg_ctl", 10)
        return joined

    class Info:
        status = "ready"

        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def is_running(self) -> bool:
            return self.ready

    states = iter([None, Info(False), Info(True)])  # no pid file yet, starting, ready
    monkeypatch.setattr(embedded_postgres, "get_server", get_server)
    monkeypatch.setattr(utils.PostmasterInfo, "read_from_pgdata", classmethod(lambda cls, p: next(states)))
    monkeypatch.setattr(embedded.time, "sleep", lambda s: None)
    assert embedded._start_or_join(tmp_path) is joined
    assert len(calls) == 2 and tmp_path.resolve() not in PostgresServer._instances


def test_a_server_that_never_gets_ready_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import embedded_postgres
    from embedded_postgres import utils

    def get_server(pgdata: Path, cleanup_mode: str) -> Any:
        raise subprocess.TimeoutExpired("pg_ctl", 10)

    monkeypatch.setattr(embedded_postgres, "get_server", get_server)
    monkeypatch.setattr(utils.PostmasterInfo, "read_from_pgdata", classmethod(lambda cls, p: None))
    with pytest.raises(subprocess.TimeoutExpired):
        embedded._start_or_join(tmp_path, wait_seconds=0)


class FakeServer:
    pgdata = Path("pgdata")


def test_a_server_that_answers_is_left_alone() -> None:
    restarts: list[Any] = []
    embedded._ensure_answering(FakeServer(), "pw", answers=lambda s, p: True, restart=restarts.append)
    assert restarts == []


def test_a_server_that_drops_connections_is_restarted_once() -> None:
    replies = iter([False, True])
    restarts: list[Any] = []
    embedded._ensure_answering(FakeServer(), "pw", answers=lambda s, p: next(replies), restart=restarts.append)
    assert len(restarts) == 1


def test_a_server_that_still_drops_connections_is_an_error() -> None:
    restarts: list[Any] = []
    with pytest.raises(embedded.EmbeddedDatabaseError, match="doesn't accept connections"):
        embedded._ensure_answering(FakeServer(), "pw", answers=lambda s, p: False, restart=restarts.append)
    assert len(restarts) == 1
