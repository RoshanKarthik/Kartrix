"""Request routing: ``handle(store, method, path, query, body)`` returns ``(status, headers, body_bytes)``.

Kept free of the HTTP server so it can be tested directly (todo/server.py serves it).
"""

from __future__ import annotations

import json
import re
from typing import Any

from todo.service import ValidationError, validate_new_todo, validate_update
from todo.storage import TodoStore

_TODO = re.compile(r"^/todos/(\d+)$")
Response = tuple[int, dict[str, str], bytes]


def _int_param(query: dict[str, str], name: str, default: int) -> int:
    try:
        value = int(query.get(name, default))
    except ValueError:
        raise ValidationError(f"{name} must be an integer") from None
    if value < 1:
        raise ValidationError(f"{name} must be at least 1")
    return value


def handle(store: TodoStore, method: str, path: str, query: dict[str, str], body: Any) -> Response:
    try:
        if path == "/health" and method == "GET":
            return 200, {"Content-Type": "application/json"}, json.dumps({"status": "ok"}).encode()
        if path == "/todos" and method == "GET":
            page = _int_param(query, "page", 1)
            per_page = min(_int_param(query, "per_page", 20), 100)
            data = {"items": store.list(page, per_page), "total": store.count(), "page": page}
            return 200, {"Content-Type": "application/json"}, json.dumps(data).encode()
        if path == "/todos" and method == "POST":
            title, tags = validate_new_todo(body)
            todo = store.create(title, tags)
            return 201, {"Content-Type": "application/json"}, json.dumps(todo).encode()
        match = _TODO.match(path)
        if match:
            todo_id = int(match.group(1))
            if method == "GET":
                todo = store.get(todo_id)
                if todo is None:
                    return 404, {"Content-Type": "application/json"}, json.dumps({"error": "not found"}).encode()
                return 200, {"Content-Type": "application/json"}, json.dumps(todo).encode()
            if method == "PATCH":
                todo = store.update(todo_id, **validate_update(body))
                if todo is None:
                    return 404, {"Content-Type": "application/json"}, json.dumps({"error": "not found"}).encode()
                return 200, {"Content-Type": "application/json"}, json.dumps(todo).encode()
            if method == "DELETE":
                if not store.delete(todo_id):
                    return 404, {"Content-Type": "application/json"}, json.dumps({"error": "not found"}).encode()
                return 204, {}, b""
        return 404, {"Content-Type": "application/json"}, json.dumps({"error": "not found"}).encode()
    except ValidationError as e:
        return 400, {"Content-Type": "application/json"}, json.dumps({"error": str(e)}).encode()
