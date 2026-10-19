"""stderr log lines (kartrix.observability.logger): short and readable on a terminal, machine format when piped."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from kartrix.config import settings
from kartrix.observability.logger import HumanFormatter, console_format


def _record(msg: str, exc: BaseException | None = None, **extra: object) -> logging.LogRecord:
    record = logging.makeLogRecord({"name": "kartrix.llm.fallback", "levelname": "WARNING", "msg": msg, **extra})
    if exc is not None:
        record.exc_info = (type(exc), exc, exc.__traceback__)
    return record


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_human_line_is_short_and_has_the_extras() -> None:
    line = HumanFormatter().format(_record("Model failed; trying the next one", model="backup", error="timed out"))
    assert line == "warning: Model failed; trying the next one (model: backup, error: timed out)"


def test_human_line_shortens_values_and_drops_tracebacks() -> None:
    try:
        raise ValueError("bad\nrequest")
    except ValueError as e:
        line = HumanFormatter().format(_record("Router failed", exc=e, error="x" * 500, status=None))
    assert "\n" not in line and "Traceback" not in line
    assert "ValueError: bad request" in line and "status" not in line
    assert len(line) < 300 and "x..." in line


def test_human_line_fits_a_legacy_console_encoding() -> None:
    line = HumanFormatter("cp1252").format(_record("Model failed → next one", model="gemma — 31b"))
    assert line == "warning: Model failed ? next one (model: gemma — 31b)"
    line.encode("cp1252")


def test_human_line_redacts_secrets() -> None:
    key = "sk-" + "a" * 40
    assert key not in HumanFormatter().format(_record(f"calling with {key}", header=f"Bearer {key}"))


def test_auto_is_human_on_a_terminal_and_the_file_format_when_piped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.logging, "console_format", "auto")
    monkeypatch.setattr(settings.logging, "format", "json")
    assert console_format(_Tty()) == "human"
    assert console_format(io.StringIO()) == "json"
    monkeypatch.setattr(settings.logging, "console_format", "json")
    assert console_format(_Tty()) == "json"


def test_setup_logging_gives_a_terminal_readable_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.observability import logger as log

    tty = _Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    monkeypatch.setattr(settings.logging, "console_format", "auto")
    monkeypatch.setattr(settings.logging, "file", None)
    app = logging.getLogger(log.ROOT_LOGGER)
    saved = list(app.handlers)
    monkeypatch.setattr(log, "_configured", False)
    app.handlers = []
    try:
        log.setup_logging()
        log.get_logger("llm.fallback").warning("Model call failed", extra={"error": "bad request"})
    finally:
        app.handlers = saved
    assert tty.getvalue() == "warning: Model call failed (error: bad request)\n"
