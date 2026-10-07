import inspect
import json

import todo.app as app


def test_helper_exists_and_builds_the_tuple():
    status, headers, body = app.json_response(201, {"a": 1})
    assert status == 201
    assert headers == {"Content-Type": "application/json"}
    assert json.loads(body) == {"a": 1}


def test_json_dumps_used_once():
    source = inspect.getsource(app)
    assert source.count("json.dumps(") == 1, "every JSON response should go through json_response"


def test_behaviour_unchanged(call):
    assert call("POST", "/todos", {"title": "x"})[0] == 201
    assert call("GET", "/todos/1")[1]["title"] == "x"
    assert call("GET", "/todos/2") == (404, {"error": "not found"})
    assert call("POST", "/todos", {})[0] == 400
    assert call("DELETE", "/todos/1") == (204, None)
