def test_pages_start_at_one(call):
    for title in ("a", "b", "c"):
        call("POST", "/todos", {"title": title})
    status, body = call("GET", "/todos", page=1, per_page=2)
    assert status == 200
    assert [t["title"] for t in body["items"]] == ["a", "b"]
    assert body["total"] == 3
    _, body = call("GET", "/todos", page=2, per_page=2)
    assert [t["title"] for t in body["items"]] == ["c"]
    _, body = call("GET", "/todos", page=3, per_page=2)
    assert body["items"] == []


def test_default_page_lists_everything(call):
    call("POST", "/todos", {"title": "only"})
    _, body = call("GET", "/todos")
    assert [t["title"] for t in body["items"]] == ["only"]
