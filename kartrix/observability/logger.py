"""Structured logging.

All ``kartrix.*`` loggers write one JSON object per line (or plain text when
``logging.format: text``) to a log file, and only WARNING+ to stderr by default
so the interactive REPL stays readable. On a terminal those stderr lines are short and
human-readable (``logging.console_format: auto``) — the full record is in the file;
piped or in a container, stderr gets ``logging.format`` like the file.
Third-party libraries are held at WARNING.

Extra fields passed via ``logger.info("msg", extra={"session_id": sid})`` are
included as top-level keys in the JSON record. Every formatted line — message, extras,
tracebacks — goes through secret redaction (``kartrix.security.secrets``) first.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kartrix.config import settings
from kartrix.security.secrets import redact

ROOT_LOGGER = "kartrix"

# Attributes every LogRecord has; anything else on the record came from `extra=`.
_RESERVED_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}

_configured = False


class JSONFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


class RedactingFormatter(logging.Formatter):
    """Plain-text formatter that redacts secrets."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class HumanFormatter(logging.Formatter):
    """One short line for a person at a terminal: ``warning: Model failed; trying the next one (model: backup,
    error: timed out)``. Extras are shortened and tracebacks reduced to the exception itself — the log file
    keeps everything. Characters the terminal's encoding lacks become ``?`` (a cp1252 console would otherwise
    turn the line into a logging error)."""

    max_value = 120

    def __init__(self, encoding: str | None = None) -> None:
        super().__init__()
        self.encoding = encoding

    def _short(self, value: Any) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= self.max_value else text[: self.max_value - 3] + "..."

    def format(self, record: logging.LogRecord) -> str:
        extras = [
            f"{key}: {self._short(value)}"
            for key, value in vars(record).items()
            if key not in _RESERVED_ATTRS and not key.startswith("_") and value is not None
        ]
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            extras.append(f"{type(exc).__name__}: {self._short(exc)}")
        line = f"{record.levelname.lower()}: {record.getMessage()}"
        if extras:
            line += f" ({', '.join(extras)})"
        line = redact(line)
        if self.encoding:
            line = line.encode(self.encoding, "replace").decode(self.encoding)
        return line


def _make_formatter(fmt: str, encoding: str | None = None) -> logging.Formatter:
    if fmt == "human":
        return HumanFormatter(encoding)
    if fmt == "json":
        return JSONFormatter()
    return RedactingFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def console_format(stream: Any = None) -> str:
    """The stderr format: ``logging.console_format``, where ``auto`` means human-readable on a terminal and
    ``logging.format`` otherwise (pipes, containers, CI — where a log collector reads the lines)."""
    cfg = settings.logging
    if cfg.console_format != "auto":
        return cfg.console_format
    stream = sys.stderr if stream is None else stream
    isatty = getattr(stream, "isatty", None)
    return "human" if callable(isatty) and isatty() else cfg.format


def setup_logging() -> None:
    """Configure handlers once; safe to call repeatedly."""
    global _configured
    if _configured:
        return
    _configured = True

    cfg = settings.logging

    # Third-party libraries (OpenAI, httpx, ...) only surface warnings.
    logging.getLogger().setLevel(logging.WARNING)

    app_logger = logging.getLogger(ROOT_LOGGER)
    app_logger.setLevel(cfg.level)
    app_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(cfg.console_level)
    console.setFormatter(_make_formatter(console_format(sys.stderr), getattr(sys.stderr, "encoding", None)))
    app_logger.addHandler(console)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_make_formatter(cfg.format))
        app_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``kartrix`` hierarchy, configuring logging on first use."""
    setup_logging()
    if name != ROOT_LOGGER and not name.startswith(ROOT_LOGGER + "."):
        name = f"{ROOT_LOGGER}.{name}"
    return logging.getLogger(name)
