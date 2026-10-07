import pytest


@pytest.mark.parametrize("title", ["   ", "\t", " \n "])
def test_whitespace_title_rejected(call, title):
    status, body = call("POST", "/todos", {"title": title})
    assert status == 400
    assert "title" in body["error"]
    _, listing = call("GET", "/todos")
    assert listing["total"] == 0


def test_title_is_trimmed(call):
    status, todo = call("POST", "/todos", {"title": "  Walk the dog  "})
    assert status == 201
    assert todo["title"] == "Walk the dog"
