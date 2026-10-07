def test_create_and_get(call):
    status, todo = call("POST", "/todos", {"title": "Buy milk", "tags": ["home"]})
    assert status == 201
    assert todo == {"id": 1, "title": "Buy milk", "done": False, "tags": ["home"]}
    status, got = call("GET", "/todos/1")
    assert status == 200 and got == todo


def test_create_requires_title(call):
    status, body = call("POST", "/todos", {"tags": []})
    assert status == 400
    assert "title" in body["error"]


def test_tags_must_be_strings(call):
    status, _ = call("POST", "/todos", {"title": "x", "tags": [1]})
    assert status == 400


def test_patch_marks_done(call):
    call("POST", "/todos", {"title": "Write report"})
    status, todo = call("PATCH", "/todos/1", {"done": True})
    assert status == 200 and todo["done"] is True


def test_delete(call):
    call("POST", "/todos", {"title": "Temporary"})
    assert call("DELETE", "/todos/1")[0] == 204
    assert call("GET", "/todos/1")[0] == 404


def test_unknown_path(call):
    assert call("GET", "/nope")[0] == 404


def test_page_must_be_positive(call):
    assert call("GET", "/todos", page=0)[0] == 400
