"""Call/import edge extraction (kartrix.context.indexers.code_graph) and the repo-map context section."""

from __future__ import annotations

from kartrix.agent import context as ctx
from kartrix.context.indexers.code_graph import Edge, extract_edges
from kartrix.context.retrievers.graph import MapEntry

PY = """import os, json as j
from app.db import session

def helper(x):
    return os.path.join(x)

class Repo:
    def save(self):
        helper(1)
        self.flush()

def main():
    Repo().save()
    print(helper(2))
"""


def _set(edges: list[Edge]) -> set[tuple[str, str, str]]:
    return {(e.source, e.kind, e.target) for e in edges}


def test_python_calls_belong_to_their_top_level_symbol() -> None:
    edges = extract_edges(PY, "app/main.py")
    assert {
        ("app/main.py", "import", "os"),
        ("app/main.py", "import", "json"),
        ("app/main.py", "import", "app.db"),
        ("helper", "call", "join"),
        ("Repo", "call", "helper"),  # a method's calls belong to its class (the indexed chunk)
        ("Repo", "call", "flush"),
        ("main", "call", "Repo"),
        ("main", "call", "save"),
        ("main", "call", "helper"),
    } <= _set(edges)
    helper_calls = [e for e in edges if e.source == "main" and e.target == "helper"]
    assert len(helper_calls) == 1 and helper_calls[0].line == 14  # deduplicated, first line kept


def test_javascript_imports_requires_and_calls() -> None:
    js = "import { a } from './lib/a';\nconst fs = require('fs');\nfunction run() { a(); new Thing(); obj.method(); }\n"
    assert _set(extract_edges(js, "src/x.js")) == {
        ("src/x.js", "import", "./lib/a"),
        ("src/x.js", "import", "fs"),
        ("run", "call", "a"),
        ("run", "call", "Thing"),
        ("run", "call", "method"),
    }


def test_go_grouped_imports_and_java_calls() -> None:
    go = 'package main\nimport (\n "fmt"\n "net/http"\n)\nfunc main() { fmt.Println("x"); serve() }\n'
    assert _set(extract_edges(go, "main.go")) == {
        ("main.go", "import", "fmt"),
        ("main.go", "import", "net/http"),
        ("main", "call", "Println"),
        ("main", "call", "serve"),
    }
    java = "import java.util.List;\nclass A { void run() { helper(); list.add(1); } }\n"
    assert {("A.java", "import", "java.util.List"), ("A", "call", "helper"), ("A", "call", "add")} <= _set(
        extract_edges(java, "A.java")
    )


def test_non_code_files_have_no_edges() -> None:
    assert extract_edges("# Title\ncall(x)\n", "README.md") == []


def test_repo_map_section_lists_most_referenced_first_within_budget() -> None:
    entries = [MapEntry(f"app/m{i}.py", f"func{i}", "function", i + 1, 10 - i) for i in range(10)]
    section = ctx.repo_map_section(entries, budget=30)
    assert section is not None and section.text.splitlines()[0] == "app/m0.py:1 function func0 — 10 refs"
    assert ctx.approx_tokens(section.text) <= 30 and section.items + section.dropped == 10
    assert ctx.repo_map_section(entries, budget=0) is None
    assert ctx.repo_map_section([], budget=30) is None
