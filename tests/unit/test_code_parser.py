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


def test_big_class_is_split_into_outline_and_methods() -> None:
    from kartrix.context.indexers.code_graph import extract_edges
    from kartrix.context.indexers.code_parser import SPLIT_CLASS_LINES

    methods = [f"    def m{i}(self):\n" + "        x = 1\n" * 8 + f"        return helper{i}()\n" for i in range(10)]
    source = 'class Big:\n    """Doc."""\n    limit = 3\n' + "".join(methods)
    assert source.count("\n") > SPLIT_CLASS_LINES
    chunks = parse_file("big.py", source)
    outline = chunks[0]
    assert (outline.name, outline.type) == ("Big", "class")
    assert "limit = 3" in outline.content and "def m3(self):" in outline.content and "x = 1" not in outline.content
    assert [c.name for c in chunks[1:]] == [f"m{i}" for i in range(10)]
    assert "return helper4()" in chunks[5].content
    owners = {e.source for e in extract_edges(source, "big.py") if e.target == "helper4"}
    assert owners == {"m4"}  # graph owners line up with the method chunks


def test_small_class_stays_one_chunk() -> None:
    chunks = parse_file("small.py", "class Small:\n    def a(self):\n        return 1\n")
    assert [(c.name, c.type) for c in chunks] == [("Small", "class")]


def test_anonymous_functions_are_named_after_binding_or_route() -> None:
    source = (
        'export const add = (a, b) => a + b;\nrouter.get("/articles", async (req, res) => {\n  res.json([]);\n});\n'
    )
    assert [c.name for c in parse_file("api.ts", source)] == ["add", "router.get /articles"]


def test_more_file_types_are_indexed() -> None:
    from kartrix.context.indexers.code_parser import is_indexable_name

    for name in ("schema.prisma", "init.sql", "index.html", "App.vue", "Dockerfile", "server.mjs", "mod.mts"):
        assert is_indexable_name(name), name
    assert not is_indexable_name("photo.png") and not is_indexable_name("LICENSE")
    assert parse_file("schema.prisma", "model User {\n  id Int @id\n}\n")[0].content.startswith("model User")
