import json

import pytest

from todo.app import handle
from todo.storage import TodoStore


@pytest.fixture
def store():
    return TodoStore()


@pytest.fixture
def call(store):
    def _call(method, path, body=None, **query):
        status, _headers, payload = handle(store, method, path, {k: str(v) for k, v in query.items()}, body)
        return status, (json.loads(payload) if payload else None)

    return _call
