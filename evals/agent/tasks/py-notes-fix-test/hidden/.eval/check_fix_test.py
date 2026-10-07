"""The fixed test file must still test fmt_note with tags (deleting the test is not a fix)."""

import ast
import pathlib
import sys

source = pathlib.Path("tests/test_formatting.py").read_text(encoding="utf-8")
names = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
ok = "test_fmt_note_with_tags" in names and "Quarterly report" in source
print("ok" if ok else "test_fmt_note_with_tags was removed or no longer checks the tagged note")
sys.exit(0 if ok else 1)
