"""Notes in a JSON file: a list of ``{"id", "text", "tags"}`` objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def default_path() -> Path:
    return Path(os.environ.get("NOTES_FILE", "notes.json"))


class NoteStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, notes: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(notes, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, text: str, tags: list[str]) -> dict[str, Any]:
        notes = self.load()
        note = {"id": max((n["id"] for n in notes), default=0) + 1, "text": text, "tags": tags}
        notes.append(note)
        self.save(notes)
        return note
