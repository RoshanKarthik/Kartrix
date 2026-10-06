"""Structured logging.

All ``kartrix.*`` loggers write one JSON object per line (or plain text when
``logging.format: text``) to a log file, and only WARNING+ to stderr by default
so the interactive REPL stays readable. Third-party libraries are held at WARNING.

Extra fields passed via ``logger.info("msg", extra={"session_id": sid})`` are
included as top-level keys in the JSON record.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kartrix.config import settings

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
        return json.dumps(payload, default=str, ensure_ascii=False)


def _make_formatter() -> logging.Formatter:
    if settings.logging.format == "json":
        return JSONFormatter()
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def setup_logging() -> None:
    """Configure handlers once; safe to call repeatedly."""
    global _configured
    if _configured:
        return
    _configured = True

    cfg = settings.logging
    formatter = _make_formatter()

    # Third-party libraries (OpenAI, httpx, ...) only surface warnings.
    logging.getLogger().setLevel(logging.WARNING)

    app_logger = logging.getLogger(ROOT_LOGGER)
    app_logger.setLevel(cfg.level)
    app_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(cfg.console_level)
    console.setFormatter(formatter)
    app_logger.addHandler(console)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``kartrix`` hierarchy, configuring logging on first use."""
    setup_logging()
    if name != ROOT_LOGGER and not name.startswith(ROOT_LOGGER + "."):
        name = f"{ROOT_LOGGER}.{name}"
    return logging.getLogger(name)
