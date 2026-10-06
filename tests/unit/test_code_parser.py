from pathlib import Path

from kartrix.context.indexers.code_parser import parse_file


def test_non_ascii_text_does_not_shift_chunks(tmp_path: Path) -> None:
    # Regression: tree-sitter byte offsets were used to slice a str, so every multi-byte
    # character before a definition shifted its name and content.
    src = '"""Ünïcödé — headers → everywhere 🚀"""\n\n\ndef first():\n    return "→"\n\n\nclass Second:\n    pass\n'
    p = tmp_path / "m.py"
    p.write_text(src, encoding="utf-8")
    chunks = parse_file(str(p))
    assert [(c.name, c.type, c.start_line) for c in chunks] == [("first", "function", 4), ("Second", "class", 8)]
    assert chunks[0].content == 'def first():\n    return "→"'
    assert chunks[1].content == "class Second:\n    pass"


def test_source_argument_skips_reading(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def on_disk():\n    pass\n", encoding="utf-8")
    assert [c.name for c in parse_file(str(p), source="def given():\n    pass\n")] == ["given"]


def test_text_files_use_overlapping_windows(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("\n".join(f"line {i}" for i in range(1, 101)), encoding="utf-8")
    chunks = parse_file(str(p))
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 50), (41, 90), (81, 100)]
