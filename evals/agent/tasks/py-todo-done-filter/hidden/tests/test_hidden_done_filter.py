def _seed(call):
    for i in range(5):
        call("POST", "/todos", {"title": f"t{i}"})
    for todo_id in (2, 4):
        call("PATCH", f"/todos/{todo_id}", {"done": True})


def test_done_true(call):
    _seed(call)
    status, body = call("GET", "/todos", done="true")
    assert status == 200
    assert [t["id"] for t in body["items"]] == [2, 4]
    assert body["total"] == 2


def test_done_false(call):
    _seed(call)
    _, body = call("GET", "/todos", done="false")
    assert [t["id"] for t in body["items"]] == [1, 3, 5]
    assert body["total"] == 3


def test_done_with_pagination(call):
    _seed(call)
    _, body = call("GET", "/todos", done="false", page=2, per_page=2)
    assert [t["id"] for t in body["items"]] == [5]
    assert body["total"] == 3


def test_done_invalid(call):
    assert call("GET", "/todos", done="maybe")[0] == 400


def test_no_filter_lists_all(call):
    _seed(call)
    _, body = call("GET", "/todos")
    assert body["total"] == 5
