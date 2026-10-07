"""Jailed file tools (B1, A6): ranged read, atomic write, exact edit, grep, glob."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kartrix.config import settings
from kartrix.security import workspace as ws_mod
from kartrix.security.workspace import set_workspace
from kartrix.tools.filesystem_tools import (
    append_file,
    compile_glob,
    delete_file,
    edit_file,
    file_exists,
    glob,
    grep,
    list_directory,
    read_file,
    write_file,
)


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    previous = ws_mod._current
    set_workspace(root)
    yield root
    ws_mod._current = previous


def call(tool: Any, **kwargs: Any) -> str:
    return tool.invoke(kwargs)


# ── read_file ─────────────────────────────────────────────────────────


def test_read_numbers_lines(root: Path) -> None:
    (root / "a.py").write_text("one\ntwo\nthree\n")
    assert call(read_file, file_path="a.py") == "     1\tone\n     2\ttwo\n     3\tthree"


def test_read_range_and_continuation_hint(root: Path) -> None:
    (root / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 101)))
    out = call(read_file, file_path="big.txt", offset=10, limit=3)
    assert out.splitlines() == [
        "    10\tline 10",
        "    11\tline 11",
        "    12\tline 12",
        "… more lines follow; continue with offset=13",
    ]


def test_read_edge_cases(root: Path) -> None:
    (root / "empty.txt").write_text("")
    (root / "short.txt").write_text("a\nb\n")
    (root / "bin.dat").write_bytes(b"\x00\x01\x02")
    (root / "latin.txt").write_bytes("café".encode("latin-1"))
    (root / "dir").mkdir()
    assert call(read_file, file_path="empty.txt") == "(empty file)"
    assert "past the end" in call(read_file, file_path="short.txt", offset=5)
    assert "binary" in call(read_file, file_path="bin.dat")
    assert "UTF-8" in call(read_file, file_path="latin.txt")
    assert "not a file" in call(read_file, file_path="dir")
    assert "not found" in call(read_file, file_path="nope.txt")
    assert ">= 1" in call(read_file, file_path="short.txt", offset=0)


def test_read_crlf_and_long_lines(root: Path) -> None:
    (root / "win.txt").write_bytes(b"a\r\nb\r\n")
    (root / "min.js").write_text("x" * 5000)
    assert call(read_file, file_path="win.txt") == "     1\ta\n     2\tb"
    assert call(read_file, file_path="min.js").endswith("[line truncated]")


def test_read_jail_errors_are_strings(root: Path, tmp_path: Path) -> None:
    (root / ".env").write_text("API_KEY=sk-secret")
    (tmp_path / "outside.txt").write_text("x")
    assert call(read_file, file_path=".env").startswith("Error: access denied")
    assert "outside the workspace" in call(read_file, file_path="../outside.txt")
    assert "outside the workspace" in call(read_file, file_path=str(tmp_path / "outside.txt"))


# ── write / append ────────────────────────────────────────────────────


def test_write_creates_parents_and_keeps_bytes_exact(root: Path) -> None:
    assert call(write_file, file_path="src/pkg/mod.py", content="a\nb\n").startswith("Created src/pkg/mod.py")
    assert (root / "src/pkg/mod.py").read_bytes() == b"a\nb\n"  # no CRLF translation on Windows
    assert call(write_file, file_path="src/pkg/mod.py", content="c").startswith("Overwrote")
    assert (root / "src/pkg/mod.py").read_text() == "c"
    assert not list(root.rglob("*.tmp"))  # temp files cleaned up


def test_write_refusals(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "d").mkdir()
    assert "protected" in call(write_file, file_path=".env", content="X=1")
    assert "protected" in call(write_file, file_path=".git/hooks/pre-commit", content="evil")
    assert "outside the workspace" in call(write_file, file_path="../x.py", content="")
    assert "is a directory" in call(write_file, file_path="d", content="")
    monkeypatch.setattr(settings.workspace, "max_file_kb", 1)
    assert "too large" in call(write_file, file_path="big.txt", content="x" * 2000)
    assert not (root / ".env").exists() and not (root / ".git").exists()


def test_append(root: Path) -> None:
    (root / "log.txt").write_text("a\n")
    assert call(append_file, file_path="log.txt", content="b\n").startswith("Appended")
    assert (root / "log.txt").read_text() == "a\nb\n"
    assert "not found" in call(append_file, file_path="missing.txt", content="x")


# ── edit_file ─────────────────────────────────────────────────────────


def test_edit_unique_match(root: Path) -> None:
    (root / "m.py").write_text("def f():\n    return 1\n\ndef g():\n    return 2\n")
    out = call(edit_file, file_path="m.py", old_string="    return 2", new_string="    return 3")
    assert out == "Edited m.py: replaced 1 occurrence (first at line 5)"
    assert (root / "m.py").read_text().endswith("return 3\n")


def test_edit_ambiguous_and_replace_all(root: Path) -> None:
    (root / "m.py").write_text("x = 1\nx = 1\n")
    assert "occurs 2 times" in call(edit_file, file_path="m.py", old_string="x = 1", new_string="x = 2")
    assert (root / "m.py").read_text() == "x = 1\nx = 1\n"  # unchanged
    assert "replaced 2 occurrences" in call(
        edit_file, file_path="m.py", old_string="x = 1", new_string="x = 2", replace_all=True
    )
    assert (root / "m.py").read_text() == "x = 2\nx = 2\n"


def test_edit_not_found_hints(root: Path) -> None:
    (root / "m.py").write_text("a = 1\nb = 2\n")
    assert "read the file again" in call(edit_file, file_path="m.py", old_string="a  = 1", new_string="x")
    assert "line-number prefixes" in call(edit_file, file_path="m.py", old_string="     1\ta = 1", new_string="x")
    assert "identical" in call(edit_file, file_path="m.py", old_string="a = 1", new_string="a = 1")
    assert "cannot be empty" in call(edit_file, file_path="m.py", old_string="", new_string="x")
    assert "not found" in call(edit_file, file_path="nope.py", old_string="a", new_string="b")


def test_edit_keeps_crlf_line_endings(root: Path) -> None:
    (root / "w.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    assert "replaced 1" in call(edit_file, file_path="w.py", old_string="a = 1\nb = 2", new_string="a = 1\nb = 20")
    assert (root / "w.py").read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"


def test_edit_protected_file(root: Path) -> None:
    (root / ".env").write_text("A=1")
    assert "protected" in call(edit_file, file_path=".env", old_string="A=1", new_string="A=2")
    assert (root / ".env").read_text() == "A=1"


# ── glob / grep ───────────────────────────────────────────────────────


@pytest.fixture
def project(root: Path) -> Path:
    files = {
        "main.py": "import os\nprint('hello')\n",
        "src/app.py": "def handler():\n    return 'Hello'\n",
        "src/ui/view.ts": "export const hello = 1;\n",
        "src/ui/view.tsx": "<Hello />\n",
        "build/out.py": "hello = 'generated'\n",
        ".env": "SECRET=hello\n",
        ".gitignore": "build/\n",
        "README.md": "Hello world\nhello again\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "logo.png").write_bytes(b"\x89PNG\x00hello")
    return root


@pytest.mark.parametrize(
    ("pattern", "path", "ok"),
    [
        ("*.py", "main.py", True),
        ("*.py", "src/app.py", False),  # "*" never crosses "/"
        ("**/*.py", "src/app.py", True),
        ("**/*.py", "main.py", True),  # "**/" also matches zero directories
        ("src/**", "src/ui/view.ts", True),
        ("src/*.{ts,tsx}", "src/a.tsx", True),
        ("test_?.py", "test_1.py", True),
        ("[!a]*.py", "app.py", False),
        ("a.py", "a_py", False),  # "." is literal
    ],
)
def test_compile_glob(pattern: str, path: str, ok: bool) -> None:
    assert bool(compile_glob(pattern).fullmatch(path)) is ok


def test_glob_respects_gitignore_and_secrets(project: Path) -> None:
    assert call(glob, pattern="**/*.py").splitlines() == ["main.py", "src/app.py"]
    assert call(glob, pattern="*.{ts,tsx}", path="src/ui").splitlines() == ["src/ui/view.ts", "src/ui/view.tsx"]
    assert ".env" not in call(glob, pattern="**/.*")
    assert call(glob, pattern="*.rs") == "No files found"
    assert "outside" in call(glob, pattern="*", path="..")


def test_grep_content_mode(project: Path) -> None:
    out = call(grep, pattern="hello")
    assert out.splitlines() == [
        "README.md:2: hello again",
        "main.py:2: print('hello')",
        "src/ui/view.ts:1: export const hello = 1;",
    ]
    assert "SECRET" not in out and "generated" not in out and "logo.png" not in out


def test_grep_options(project: Path) -> None:
    assert call(grep, pattern="hello", ignore_case=True, output_mode="files").splitlines() == [
        "README.md",
        "main.py",
        "src/app.py",
        "src/ui/view.ts",
        "src/ui/view.tsx",
    ]
    assert call(grep, pattern="hello", ignore_case=True, output_mode="count", glob="*.md") == "README.md: 2"
    assert call(grep, pattern="def \\w+", path="src/app.py") == "src/app.py:1: def handler():"
    assert call(grep, pattern="hello", glob="src/**/*.ts") == "src/ui/view.ts:1: export const hello = 1;"
    limited = call(grep, pattern="hello", ignore_case=True, max_results=2).splitlines()
    assert len(limited) == 3 and limited[-1].startswith("… stopped at 2")
    assert "invalid regular expression" in call(grep, pattern="(")
    assert call(grep, pattern="zzz") == "No matches found"


def test_grep_never_reads_protected_file_directly(project: Path) -> None:
    assert "protected" in call(grep, pattern="SECRET", path=".env")


def test_list_and_exists(project: Path) -> None:
    listing = call(list_directory).splitlines()
    assert "src/" in listing and "main.py" in listing
    assert call(file_exists, file_path="src/app.py") == "True"
    assert call(file_exists, file_path="src/nope.py") == "False"
    assert "outside" in call(list_directory, directory="..")


# ── delete_file ───────────────────────────────────────────────────────


def test_delete_file(root: Path) -> None:
    (root / "scratch.py").write_text("x")
    (root / "src").mkdir()
    (root / ".env").write_text("SECRET=1")
    assert call(delete_file, file_path="scratch.py") == "Deleted scratch.py"
    assert not (root / "scratch.py").exists()
    assert "does not exist" in call(delete_file, file_path="scratch.py")
    assert "is a directory" in call(delete_file, file_path="src") and (root / "src").is_dir()
    assert call(delete_file, file_path=".env").startswith("Error") and (root / ".env").exists()  # protected
    assert call(delete_file, file_path="../outside.txt").startswith("Error")  # jailed
