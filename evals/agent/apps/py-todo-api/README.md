# todo-api

A small JSON API for todos — Python standard library only (`http.server` + `sqlite3`).

```
python -m todo.server            # serves on http://127.0.0.1:8000
python -m pytest -q              # tests
```

| Method | Path            | Body / query                         |
|--------|-----------------|--------------------------------------|
| GET    | /todos          | `?page=1&per_page=20`                |
| POST   | /todos          | `{"title": "...", "tags": ["..."]}`  |
| GET    | /todos/{id}     |                                      |
| PATCH  | /todos/{id}     | `{"title"?, "done"?, "tags"?}`       |
| DELETE | /todos/{id}     |                                      |

Pages start at 1.
