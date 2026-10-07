from todo.storage import TodoStore


def _store_with_todo():
    store = TodoStore()
    store.create("Original", ["a", "b"])
    return store


def test_update_title_keeps_other_fields():
    store = _store_with_todo()
    todo = store.update(1, title="Renamed")
    assert todo == {"id": 1, "title": "Renamed", "done": False, "tags": ["a", "b"]}


def test_mark_done_and_undone():
    store = _store_with_todo()
    assert store.update(1, done=True)["done"] is True
    assert store.get(1)["done"] is True
    assert store.update(1, done=False)["done"] is False
    assert store.get(1)["done"] is False


def test_replace_tags():
    store = _store_with_todo()
    store.update(1, tags=["x"])
    assert store.get(1)["tags"] == ["x"]


def test_unknown_id():
    assert TodoStore().update(99, title="x") is None
