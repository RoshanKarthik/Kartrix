"""How notes are shown in the terminal."""

from __future__ import annotations

from typing import Any


def normalise_tags(tags: list[str]) -> list[str]:
    """Tags lower-cased, without duplicates, sorted alphabetically."""
    return sorted({t.strip().lower() for t in tags if t.strip()})


def format_note(note: dict[str, Any]) -> str:
    """One line per note: ``#<id> [tag, tag] text`` (no brackets when there are no tags)."""
    tags = normalise_tags(note.get("tags", []))
    label = f" [{', '.join(tags)}]" if tags else ""
    return f"#{note['id']}{label} {note['text']}"
