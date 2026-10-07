from todo.storage import TodoStore


def test_create_returns_todo():
    store = TodoStore()
    todo = store.create("Read a book", ["fun"])
    assert todo["title"] == "Read a book"
    assert todo["done"] is False
    assert todo["tags"] == ["fun"]


def test_count():
    store = TodoStore()
    store.create("a")
    store.create("b")
    assert store.count() == 2


def test_delete_missing():
    assert TodoStore().delete(42) is False
