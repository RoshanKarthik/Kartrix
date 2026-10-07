import pathlib

import notes.formatting as formatting


def test_new_name():
    assert formatting.format_note({"id": 1, "text": "x", "tags": ["B", "a"]}) == "#1 [a, b] x"


def test_old_name_gone():
    assert not hasattr(formatting, "fmt_note")
    here = pathlib.Path(__file__).resolve()
    root = here.parent.parent
    left = [
        str(p.relative_to(root))
        for p in [*root.glob("notes/*.py"), *root.glob("tests/*.py"), root / "README.md"]
        if p.is_file() and p.resolve() != here and "fmt_note" in p.read_text(encoding="utf-8")
    ]
    assert not left, f"fmt_note still used in {left}"
