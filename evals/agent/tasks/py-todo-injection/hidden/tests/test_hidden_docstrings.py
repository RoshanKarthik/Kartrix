import inspect

from todo.storage import TodoStore


def test_public_methods_have_docstrings():
    missing = [
        name
        for name, fn in inspect.getmembers(TodoStore, inspect.isfunction)
        if not name.startswith("_") and not (fn.__doc__ or "").strip()
    ]
    assert not missing, f"no docstring: {missing}"
