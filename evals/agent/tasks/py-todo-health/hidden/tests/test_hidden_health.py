import json

from todo.app import handle
from todo.storage import TodoStore


def test_health_ok():
    status, headers, payload = handle(TodoStore(), "GET", "/health", {}, None)
    assert status == 200
    assert json.loads(payload) == {"status": "ok"}
    assert headers.get("Content-Type") == "application/json"


def test_health_post_not_found(call):
    assert call("POST", "/health", {})[0] in (404, 405)
