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
_NOT_FOUND = {"error": "not found"}


def json_response(status: int, payload: Any) -> Response:
    return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


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
        if path == "/todos" and method == "GET":
            page = _int_param(query, "page", 1)
            per_page = min(_int_param(query, "per_page", 20), 100)
            return json_response(200, {"items": store.list(page, per_page), "total": store.count(), "page": page})
        if path == "/todos" and method == "POST":
            title, tags = validate_new_todo(body)
            return json_response(201, store.create(title, tags))
        match = _TODO.match(path)
        if match:
            todo_id = int(match.group(1))
            if method == "GET":
                todo = store.get(todo_id)
                return json_response(404, _NOT_FOUND) if todo is None else json_response(200, todo)
            if method == "PATCH":
                todo = store.update(todo_id, **validate_update(body))
                return json_response(404, _NOT_FOUND) if todo is None else json_response(200, todo)
            if method == "DELETE":
                if not store.delete(todo_id):
                    return json_response(404, _NOT_FOUND)
                return 204, {}, b""
        return json_response(404, _NOT_FOUND)
    except ValidationError as e:
        return json_response(400, {"error": str(e)})
