from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter_languages import get_parser

from kartrix.observability.logger import get_logger

logger = get_logger(__name__)


# Maps file extension → tree-sitter language name.
# tree-sitter-languages bundles all these grammars, no extra installs needed.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
}

# Text/config files (and languages without a bundled grammar, e.g. Prisma, SQL, Vue) have no AST
# here — they are chunked by line count instead.
TEXT_EXTENSIONS = {
    ".md", ".mdx", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".xml",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".astro",
    ".sql", ".prisma", ".graphql", ".gql", ".proto", ".tf",
}  # fmt: skip
TEXT_FILENAMES = {"Dockerfile", "Makefile", "Procfile", "Containerfile", "Justfile"}  # no extension
ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS


def is_indexable_name(path: str | Path) -> bool:
    """A file type the index understands (by extension, or by name for Dockerfile, Makefile, …)."""
    p = Path(path)
    return p.suffix.lower() in ALL_EXTENSIONS or p.name in TEXT_FILENAMES


# Sliding window settings for text files and AST fallback.
# CHUNK_OVERLAP ensures context isn't lost at chunk boundaries.
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10

# Tree-sitter node type names that correspond to a named, indexable block.
# These are consistent across languages — tree-sitter uses the same names where possible.
BLOCK_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "arrow_function",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",
    "interface_declaration",
    "function_item",  # Rust uses this instead of function_definition
    "func_declaration",  # Go
}

# A class longer than this is indexed as an outline (header + method signatures) plus one chunk per
# method: one embedding of a 500-line class is truncated and matches nothing in particular.
SPLIT_CLASS_LINES = 80


def is_split_class(node: Any) -> bool:
    """True for a class block that is indexed method by method (the code graph uses the same rule, so
    edge owners and chunk names agree)."""
    return "class" in node.type and node.end_point[0] - node.start_point[0] + 1 > SPLIT_CLASS_LINES


@dataclass
class ParsedChunk:
    name: str
    type: str  # "function" | "class" | "block"
    content: str
    source: str  # absolute path to the file
    start_line: int
    end_line: int


def parse_file(filepath: str, source: str | None = None) -> list[ParsedChunk]:
    """Entry point — routes to AST parsing or sliding window based on file type.

    Pass ``source`` when the caller has already read the file (avoids a second read).
    """
    ext = Path(filepath).suffix.lower()
    if source is None:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")

    if ext in TEXT_EXTENSIONS or Path(filepath).name in TEXT_FILENAMES:
        return _sliding_window(source.splitlines(), filepath)

    language_name = EXTENSION_TO_LANGUAGE.get(ext)
    if not language_name:
        raise ValueError(f"Unsupported file type: {ext}")

    return _parse_with_treesitter(source, filepath, language_name)


def _parse_with_treesitter(source: str, filepath: str, language_name: str) -> list[ParsedChunk]:
    """Parse source with the appropriate tree-sitter grammar and extract named blocks."""
    logger.info(f"Parsing {language_name} file: {filepath}")
    parser = get_parser(language_name)
    # tree-sitter offsets are byte offsets into the UTF-8 encoding, so slice bytes, not
    # the str — otherwise any non-ASCII character shifts every later name and chunk.
    data = source.encode("utf-8")
    tree = parser.parse(data)
    lines = source.splitlines()

    chunks: list[ParsedChunk] = []
    _walk(tree.root_node, data, filepath, chunks, depth=0)

    # If the AST yielded nothing (e.g. a file with only imports), fall back to line chunks
    if not chunks:
        logger.info(f"No AST blocks found in {filepath}, falling back to sliding window")
        return _sliding_window(lines, filepath)

    logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
    return chunks


def _walk(node, source: bytes, filepath: str, chunks: list, depth: int):
    """
    Recursively walk the AST. When a named block node is found, record it and stop
    descending — this keeps chunks at the top level and avoids duplicating nested functions.
    """
    if node.type in BLOCK_NODE_TYPES and is_split_class(node):
        _split_class(node, source, filepath, chunks)
        return
    if node.type in BLOCK_NODE_TYPES:
        name = _extract_name(node, source)
        content = source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
        chunk_type = "class" if "class" in node.type else "function"
        chunks.append(
            ParsedChunk(
                name=name,
                type=chunk_type,
                content=content,
                source=filepath,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
        logger.debug(f"  Found {chunk_type} '{name}' (lines {node.start_point[0] + 1}-{node.end_point[0] + 1})")
        return  # stop here — don't index nested functions/classes as separate chunks

    for child in node.children:
        _walk(child, source, filepath, chunks, depth + 1)


def _methods(node: Any) -> list[Any]:
    """The block nodes directly inside a class (its methods, nested classes), not their insides."""
    found: list[Any] = []

    def visit(n: Any) -> None:
        for child in n.children:
            if child.type in BLOCK_NODE_TYPES:
                found.append(child)
            else:
                visit(child)

    visit(node)
    return found


def _split_class(node: Any, source: bytes, filepath: str, chunks: list) -> None:
    """A big class: an outline chunk (the class without its method bodies — fields, docstring and every
    method's first line) and a chunk per method."""
    name = _extract_name(node, source)
    methods = _methods(node)
    text = source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore").splitlines()
    first = node.start_point[0]
    hidden: set[int] = set()
    for m in methods:  # keep each method's first line, hide the rest of its body
        hidden.update(range(m.start_point[0] + 1 - first, m.end_point[0] + 1 - first))
    outline = [line for i, line in enumerate(text) if i not in hidden]
    chunks.append(
        ParsedChunk(name, "class", "\n".join(outline), filepath, node.start_point[0] + 1, node.end_point[0] + 1)
    )
    for m in methods:
        if is_split_class(m):
            _split_class(m, source, filepath, chunks)
            continue
        chunks.append(
            ParsedChunk(
                name=_extract_name(m, source),
                type="class" if "class" in m.type else "function",
                content=source[m.start_byte : m.end_byte].decode("utf-8", errors="ignore"),
                source=filepath,
                start_line=m.start_point[0] + 1,
                end_line=m.end_point[0] + 1,
            )
        )


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _extract_name(node, source: bytes) -> str:
    """Find the identifier child of a block node and return its text as the chunk name. Anonymous
    functions are named after what they are bound to: ``const add = () => …`` → ``add``, a handler
    passed to ``router.get("/users", …)`` → ``router.get /users``."""
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return _node_text(child, source)
    parent = getattr(node, "parent", None)
    if parent is not None and parent.type in ("variable_declarator", "assignment_expression", "pair"):
        for child in parent.children:
            if child.type in ("identifier", "property_identifier", "member_expression", "string"):
                return _node_text(child, source).strip("'\"")
    if parent is not None and parent.type == "arguments" and parent.parent is not None:
        call = parent.parent
        callee = call.child_by_field_name("function") if hasattr(call, "child_by_field_name") else None
        route = next((c for c in parent.children if c.type in ("string", "template_string")), None)
        if callee is not None:
            label = _node_text(callee, source)
            return f"{label} {_node_text(route, source).strip('`\'"')}" if route is not None else label
    return node.type  # fallback to node type if no name found


def _sliding_window(lines: list[str], filepath: str) -> list[ParsedChunk]:
    """
    Split lines into overlapping fixed-size chunks.
    Used for text files and as a fallback when AST parsing finds nothing.
    """
    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(lines), step)):
        end = min(start + CHUNK_SIZE, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(
                ParsedChunk(
                    name=f"chunk_{i}",
                    type="block",
                    content=text,
                    source=filepath,
                    start_line=start + 1,
                    end_line=end,
                )
            )
        if end == len(lines):
            break

    logger.info(f"Parsed {len(chunks)} chunks from {filepath}")
    return chunks
