"""SQLite storage for todos."""

from __future__ import annotations

import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    tags TEXT NOT NULL DEFAULT ''
)
"""


def _row_to_todo(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "done": bool(row["done"]),
        "tags": [t for t in row["tags"].split(",") if t],
    }


class TodoStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)

    def create(self, title: str, tags: list[str] | None = None) -> dict[str, Any]:
        cur = self.conn.execute("INSERT INTO todos (title, tags) VALUES (?, ?)", (title, ",".join(tags or [])))
        self.conn.commit()
        todo = self.get(cur.lastrowid or 0)
        assert todo is not None
        return todo

    def get(self, todo_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        return _row_to_todo(row) if row else None

    def list(self, page: int = 1, per_page: int = 20) -> list[dict[str, Any]]:
        offset = (page - 1) * per_page
        rows = self.conn.execute("SELECT * FROM todos ORDER BY id LIMIT ? OFFSET ?", (per_page, offset)).fetchall()
        return [_row_to_todo(r) for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM todos").fetchone()[0])

    def update(self, todo_id: int, **fields: Any) -> dict[str, Any] | None:
        if self.get(todo_id) is None:
            return None
        if "title" in fields:
            self.conn.execute("UPDATE todos SET title = ? WHERE id = ?", (fields["title"], todo_id))
        if "done" in fields:
            self.conn.execute("UPDATE todos SET done = ? WHERE id = ?", (1 if fields["done"] else 0, todo_id))
        self.conn.commit()
        return self.get(todo_id)

    def delete(self, todo_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        self.conn.commit()
        return cur.rowcount > 0
