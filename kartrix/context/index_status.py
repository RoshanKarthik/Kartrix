"""Whether the code index is ready — it is built in the background so startup never waits for it.

``search_codebase`` adds a note while the index is incomplete, so the model falls back to
grep / glob / read_file instead of trusting a partial result.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

State = Literal["unknown", "building", "ready", "failed"]


@dataclass(frozen=True)
class IndexStatus:
    state: State
    detail: str | None = None
    since: float = 0.0


_status = IndexStatus("unknown")
_lock = threading.Lock()


def set_status(state: State, detail: str | None = None) -> None:
    global _status
    with _lock:
        _status = IndexStatus(state, detail, time.time())


def get_status() -> IndexStatus:
    return _status


def search_note() -> str | None:
    """A line for search results while the index can't be trusted to be complete."""
    status = _status
    if status.state == "building":
        return (
            "Note: the code index is still being built, so these results may be incomplete — "
            "use grep / glob / read_file to be sure."
        )
    if status.state == "failed":
        return f"Note: the code index could not be updated ({status.detail}); use grep / glob / read_file to be sure."
    return None
