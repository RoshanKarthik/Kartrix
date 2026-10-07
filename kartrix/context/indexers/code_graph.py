"""The code graph (GraphRAG for code): call and import edges extracted with tree-sitter.

Every edge says *who* (``source``: the top-level function/class the code sits in — the same name as its
indexed chunk — or the file path for module-level code) *calls* or *imports* *what* (``target``: the
called name, or the imported module/path) at which ``line``. Edges are stored in ``code_edges`` at
indexing time and used to:

- expand search results along the graph (``kartrix.context.retrievers.pg_hybrid.graph_neighbors``),
- answer "who calls X / what does X call" (the ``symbol_graph`` tool),
- rank symbols by how often they are referenced (the repo map in ``kartrix.agent.context``).

Names are resolved by text only (no type inference): ``obj.save()`` is an edge to ``save``. Builtins
and library calls are stored too; they simply never match an indexed chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter_languages import get_parser

from kartrix.context.indexers.code_parser import BLOCK_NODE_TYPES, EXTENSION_TO_LANGUAGE, _extract_name

CALL_TYPES = {
    "call",  # python
    "call_expression",  # js/ts, go, rust, c/c++, kotlin, swift
    "method_invocation",  # java
    "invocation_expression",  # c#
    "new_expression",  # js/ts: new Foo()
    "object_creation_expression",  # java/c#: new Foo()
    "method_call",  # ruby
}
IMPORT_TYPES = {
    "import_statement",  # python, js/ts
    "import_from_statement",  # python
    "import_declaration",  # java
    "import_spec",  # go
    "use_declaration",  # rust
    "preproc_include",  # c/c++
    "using_directive",  # c#
}
_CALLEE_FIELDS = ("function", "constructor", "name", "method", "type")
_MEMBER_FIELDS = ("attribute", "property", "field", "name")
_IMPORT_FIELDS = ("module_name", "source", "path", "argument")
_MAX_TARGET = 200


@dataclass(frozen=True)
class Edge:
    source: str
    kind: str  # "call" | "import"
    target: str
    line: int


def supports(path: str | Path) -> bool:
    return Path(path).suffix.lower() in EXTENSION_TO_LANGUAGE


def extract_edges(source: str, rel: str) -> list[Edge]:
    """Call and import edges of one file (``rel`` is its repo-relative path). Deduplicated: one edge
    per (source, kind, target), at its first line."""
    language = EXTENSION_TO_LANGUAGE.get(Path(rel).suffix.lower())
    if language is None:
        return []
    data = source.encode("utf-8")
    tree = get_parser(language).parse(data)
    found: dict[tuple[str, str, str], Edge] = {}
    _walk(tree.root_node, data, rel, rel, found)
    return list(found.values())


def _walk(node: Any, data: bytes, rel: str, owner: str, found: dict[tuple[str, str, str], Edge]) -> None:
    if node.type in BLOCK_NODE_TYPES and owner == rel:  # top-level block: its name owns what's inside
        owner = _extract_name(node, data)
    targets: list[tuple[str, str]] = []
    if node.type in CALL_TYPES:
        callee = _callee(node, data)
        if callee == "require":  # js: require('x') is an import
            arg = _first_string(node, data)
            targets.append(("import", arg) if arg else ("call", callee))
        elif callee:
            targets.append(("call", callee))
    elif node.type in IMPORT_TYPES:
        targets += [("import", t) for t in _imports(node, data)]
    for kind, target in targets:
        target = target.strip()[:_MAX_TARGET]
        if target:
            found.setdefault((owner, kind, target), Edge(owner, kind, target, node.start_point[0] + 1))
    for child in node.children:
        _walk(child, data, rel, owner, found)


def _text(node: Any, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _field(node: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    return None


def _rightmost_name(node: Any, data: bytes) -> str:
    """``foo`` → foo; ``a.b.save`` → save; ``Foo::new`` → new; ``obj.method`` → method."""
    for _ in range(20):  # bounded descent
        if node.type.endswith("identifier") or node.type in ("name", "constant", "type_identifier"):
            return _text(node, data)
        nxt = _field(node, _MEMBER_FIELDS)
        if nxt is None:
            named = [c for c in node.named_children if c.type not in ("arguments", "argument_list")]
            if not named:
                return ""
            nxt = named[-1]
        node = nxt
    return ""


def _callee(node: Any, data: bytes) -> str:
    target = _field(node, _CALLEE_FIELDS)
    if target is None:
        named = node.named_children
        if not named:
            return ""
        target = named[0]
    return _rightmost_name(target, data)


def _first_string(node: Any, data: bytes) -> str:
    stack = list(node.children)
    while stack:
        child = stack.pop(0)
        if "string" in child.type:
            return _unquote(_text(child, data))
        stack.extend(child.children)
    return ""


def _unquote(text: str) -> str:
    return text.strip().strip("\"'`<>")


def _imports(node: Any, data: bytes) -> list[str]:
    if node.type == "import_statement" and node.child_by_field_name("source") is None:  # python: import a, b as c
        names = []
        for child in node.named_children:
            if child.type == "aliased_import":
                child = child.child_by_field_name("name") or child
            if child.type == "dotted_name":
                names.append(_text(child, data))
        return names
    target = _field(node, _IMPORT_FIELDS)
    if target is None:
        named = [c for c in node.named_children if c.type != "comment"]
        if not named:
            return []
        target = named[0]
    if target.type in ("import_spec", "import_spec_list"):  # go: import ( ... ) — each spec is its own node
        return []
    return [_unquote(_text(target, data))]
