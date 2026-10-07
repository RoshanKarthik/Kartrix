# notes

A tiny command-line notebook (Python standard library only). Notes live in a JSON file.

```
python -m notes add "Call the plumber" --tag home
python -m notes list [--tag home]
python -m notes stats
python -m pytest -q
```

`NOTES_FILE` sets where notes are stored (default `notes.json`).
