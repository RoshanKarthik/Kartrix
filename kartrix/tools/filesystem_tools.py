"""File tools for the agents, all jailed to the workspace (B1) — see ``kartrix.security.workspace``.

Paths are relative to the workspace root (absolute paths inside it also work).
Errors come back as ``"Error: ..."`` strings so the model can recover.
Writes are atomic (temp file + ``os.replace``): a crash never leaves a half-written file,
and a write never goes *through* a symlink or hard link to a file elsewhere.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

from langchain.tools import tool

from kartrix.config import settings
from kartrix.context.discovery import RepoFilter, iter_files
from kartrix.observability.logger import get_logger
from kartrix.security.permissions import PermissionDeniedError, ensure_writes_allowed
from kartrix.security.secrets import contains_placeholder, redact
from kartrix.security.workspace import CASE_INSENSITIVE, WorkspaceError, get_workspace

logger = get_logger(__name__)

_DEFAULT_READ_LINES = 2000
_MAX_LINE_CHARS = 2000  # longer lines are cut in tool output (minified files)
_MAX_OUTPUT_CHARS = 100_000
_MAX_LIST_ENTRIES = 500
_MAX_GLOB_RESULTS = 500
_GREP_DEADLINE_S = 15.0
_GREP_SCAN_CHARS = 10_000  # only the start of very long lines is searched (bounds regex cost)
_BINARY_SNIFF_BYTES = 8192


def _max_bytes() -> int:
    return settings.workspace.max_file_kb * 1024


def _tool_errors[**P](fn: Callable[P, str]) -> Callable[P, str]:
    """Turn jail violations and OS errors into ``Error: ...`` strings for the model."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except (WorkspaceError, PermissionDeniedError) as e:
            return f"Error: {e}"
        except PermissionError:
            return "Error: permission denied by the operating system"
        except OSError as e:
            return f"Error: {e.strerror or e}"

    return wrapper


def _is_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        return b"\x00" in fh.read(_BINARY_SNIFF_BYTES)


def _clip(line: str) -> str:
    return line if len(line) <= _MAX_LINE_CHARS else line[:_MAX_LINE_CHARS] + " … [line truncated]"


def _atomic_write(path: Path, data: bytes) -> None:
    ws = get_workspace()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after mkdir: the parent must still really be inside the workspace.
    ws.resolve(str(path.parent), "write")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".kartrix-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _load_text(path: Path, rel: str) -> str:
    """Whole file as text with line endings untouched (for edits)."""
    if not path.is_file():
        raise WorkspaceError(f"file not found: {rel}")
    size = path.stat().st_size
    if size > _max_bytes():
        raise WorkspaceError(f"file too large to edit ({size} bytes, max {_max_bytes()})")
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError(f"file is not valid UTF-8 text: {rel}") from None


# ── reading ───────────────────────────────────────────────────────────


@tool(parse_docstring=True)
@_tool_errors
def read_file(file_path: str, offset: int = 1, limit: int = _DEFAULT_READ_LINES) -> str:
    """Read a text file. Lines come back numbered as "<number><TAB><text>"; the number prefix
    is not part of the file. Large files: read them in pieces with offset and limit.

    Args:
        file_path: Path relative to the workspace root.
        offset: First line to return (1-based).
        limit: Maximum number of lines to return (at most 2000).
    """
    ws = get_workspace()
    path = ws.resolve(file_path, "read")
    rel = ws.relative(path)
    if not path.exists():
        return f"Error: file not found: {rel}"
    if not path.is_file():
        return f"Error: not a file: {rel} (use list_directory)"
    if offset < 1 or limit < 1:
        return "Error: offset and limit must be >= 1"
    limit = min(limit, _DEFAULT_READ_LINES)
    if _is_binary(path):
        return f"Error: binary file: {rel}"

    out: list[str] = []
    chars = 0
    last = offset - 1
    more = False
    with path.open(encoding="utf-8", errors="strict") as fh:
        try:
            for n, line in enumerate(fh, 1):
                if n < offset:
                    continue
                if n >= offset + limit or chars > _MAX_OUTPUT_CHARS:
                    more = True
                    break
                text = f"{n:>6}\t{_clip(line.rstrip('\r\n'))}"
                out.append(text)
                chars += len(text) + 1
                last = n
        except UnicodeDecodeError:
            return f"Error: file is not valid UTF-8 text: {rel}"
    if not out:
        return "(empty file)" if offset == 1 else f"Error: offset {offset} is past the end of the file ({last} lines)"
    if more:
        out.append(f"… more lines follow; continue with offset={last + 1}")
    return "\n".join(out)


@tool(parse_docstring=True)
@_tool_errors
def list_directory(directory: str = ".") -> str:
    """List the entries of a directory; subdirectories end with "/".

    Args:
        directory: Path relative to the workspace root ("." for the root).
    """
    ws = get_workspace()
    path = ws.resolve(directory, "read")
    rel = ws.relative(path)
    if not path.exists():
        return f"Error: directory not found: {rel}"
    if not path.is_dir():
        return f"Error: not a directory: {rel}"
    entries = sorted(e.name + ("/" if e.is_dir(follow_symlinks=False) else "") for e in os.scandir(path))
    if not entries:
        return "(empty directory)"
    extra = len(entries) - _MAX_LIST_ENTRIES
    lines = entries[:_MAX_LIST_ENTRIES]
    if extra > 0:
        lines.append(f"… {extra} more entries (use glob to narrow down)")
    return "\n".join(lines)


@tool(parse_docstring=True)
@_tool_errors
def file_exists(file_path: str) -> str:
    """Check whether a file or directory exists. Returns "True" or "False".

    Args:
        file_path: Path relative to the workspace root.
    """
    return str(get_workspace().resolve(file_path, "read").exists())


# ── writing ───────────────────────────────────────────────────────────


@tool(parse_docstring=True)
@_tool_errors
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with the given content; parent directories are created.
    To change part of an existing file, prefer edit_file.

    Args:
        file_path: Path relative to the workspace root.
        content: The complete new file content.
    """
    ensure_writes_allowed()
    ws = get_workspace()
    path = ws.resolve(file_path, "write")
    rel = ws.relative(path)
    if path.is_dir():
        return f"Error: {rel} is a directory"
    data = content.encode("utf-8")
    if len(data) > _max_bytes():
        return f"Error: content too large ({len(data)} bytes, max {_max_bytes()})"
    existed = path.exists()
    if existed and contains_placeholder(content):
        old = _load_text(path, rel)
        if redact(old) != old:  # the model only ever saw this file with its secrets masked
            return (
                f"Error: {rel} contains secrets that were shown to you as [REDACTED:...]; overwriting it would "
                "destroy them. Use edit_file on the other parts instead."
            )
    _atomic_write(path, data)
    return f"{'Overwrote' if existed else 'Created'} {rel} ({len(data)} bytes)"


@tool(parse_docstring=True)
@_tool_errors
def append_file(file_path: str, content: str) -> str:
    """Append content to the end of an existing file.

    Args:
        file_path: Path relative to the workspace root.
        content: Text to add at the end of the file.
    """
    ensure_writes_allowed()
    ws = get_workspace()
    path = ws.resolve(file_path, "write")
    rel = ws.relative(path)
    data = (_load_text(path, rel) + content).encode("utf-8")
    if len(data) > _max_bytes():
        return f"Error: file would become too large ({len(data)} bytes, max {_max_bytes()})"
    _atomic_write(path, data)
    return f"Appended {len(content.encode('utf-8'))} bytes to {rel}"


_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\t", re.MULTILINE)


@tool(parse_docstring=True)
@_tool_errors
def delete_file(file_path: str) -> str:
    """Delete one file, e.g. a temporary helper script you created. Folders are not deleted.
    The deletion can be undone with /undo like any other change.

    Args:
        file_path: Path relative to the workspace root.
    """
    ensure_writes_allowed()
    ws = get_workspace()
    path = ws.resolve(file_path, "write")
    rel = ws.relative(path)
    if path.is_dir() and not path.is_symlink():
        return f"Error: {rel} is a directory; only files can be deleted"
    if not path.exists() and not path.is_symlink():
        return f"Error: {rel} does not exist"
    path.unlink()  # a link is removed itself, never its target
    return f"Deleted {rel}"


@tool(parse_docstring=True)
@_tool_errors
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact piece of text in a file. old_string must match the file exactly
    (whitespace and indentation included) and be unique, unless replace_all is true.
    Include a few surrounding lines to make it unique. Read the file first.

    Args:
        file_path: Path relative to the workspace root.
        old_string: Exact text to find (without read_file's line-number prefixes).
        new_string: Replacement text.
        replace_all: Replace every occurrence instead of requiring exactly one.
    """
    ensure_writes_allowed()
    ws = get_workspace()
    path = ws.resolve(file_path, "write")
    rel = ws.relative(path)
    if not old_string:
        return "Error: old_string cannot be empty (use write_file to create a file)"
    if old_string == new_string:
        return "Error: old_string and new_string are identical"
    text = _load_text(path, rel)

    if contains_placeholder(new_string) and not contains_placeholder(old_string):
        return "Error: new_string contains a [REDACTED:...] placeholder — redacted secrets can't be written back"
    count = text.count(old_string)
    if count == 0 and "\r\n" in text and "\n" in old_string and "\r\n" not in old_string:
        # The model writes "\n"; the file uses Windows line endings. Keep the file's style.
        old_string, new_string = old_string.replace("\n", "\r\n"), new_string.replace("\n", "\r\n")
        count = text.count(old_string)
    if count == 0 and contains_placeholder(old_string):
        return (
            f"Error: old_string includes a redacted secret ([REDACTED:...]) that isn't literally in {rel}; "
            "edit around that line instead of including it"
        )
    if count == 0:
        hint = (
            " — remove the line-number prefixes copied from read_file"
            if _LINE_NUMBER_PREFIX.search(old_string)
            else " — it must match exactly, including whitespace; read the file again"
        )
        return f"Error: old_string not found in {rel}{hint}"
    if count > 1 and not replace_all:
        return (
            f"Error: old_string occurs {count} times in {rel}; add surrounding context to make it unique, "
            "or set replace_all=true"
        )

    line = text[: text.index(old_string)].count("\n") + 1
    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    data = new_text.encode("utf-8")
    if len(data) > _max_bytes():
        return f"Error: file would become too large ({len(data)} bytes, max {_max_bytes()})"
    _atomic_write(path, data)
    n = count if replace_all else 1
    return f"Edited {rel}: replaced {n} occurrence{'s' if n > 1 else ''} (first at line {line})"


# ── searching ─────────────────────────────────────────────────────────


def _expand_braces(pattern: str) -> list[str]:
    """``a.{ts,tsx}`` → ``["a.ts", "a.tsx"]`` (no nesting)."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    head, tail = pattern[: m.start()], pattern[m.end() :]
    return [x for alt in m.group(1).split(",") for x in _expand_braces(head + alt + tail)]


def _glob_regex(pattern: str) -> str:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        at_segment_start = i == 0 or pattern[i - 1] == "/"
        if at_segment_start and pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif at_segment_start and pattern[i:] == "**":
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and (end := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1 : end]
            body = "^" + body[1:] if body.startswith("!") else body
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a glob (``*``, ``?``, ``[...]``, ``**``, ``{a,b}``) matched against the whole
    relative POSIX path."""
    pattern = pattern.strip().replace("\\", "/").removeprefix("./")
    regex = "|".join(_glob_regex(p) for p in _expand_braces(pattern))
    return re.compile(f"(?:{regex})", re.IGNORECASE if CASE_INSENSITIVE else 0)


def _searchable_files(path: Path) -> Iterator[Path]:
    """Files under ``path`` that .gitignore/config don't exclude and the jail lets us read."""
    ws = get_workspace()
    if path.is_file():
        yield path
        return
    for f in iter_files(RepoFilter.load(ws.root), path):
        if ws.is_readable(f):
            yield f


@tool(parse_docstring=True)
@_tool_errors
def glob(pattern: str, path: str = ".") -> str:
    """Find files by name pattern, e.g. "**/*.py", "src/**/*.{ts,tsx}", "tests/test_*.py".
    "*" stays inside one directory; "**/" spans any number of directories. Files ignored
    by .gitignore are skipped. Returns paths relative to the workspace root.

    Args:
        pattern: Glob pattern, relative to path.
        path: Directory to search in, relative to the workspace root.
    """
    ws = get_workspace()
    base = ws.resolve(path, "read")
    if not base.is_dir():
        return f"Error: not a directory: {ws.relative(base)}"
    if not pattern.strip():
        return "Error: pattern cannot be empty"
    rx = compile_glob(pattern)
    matches = [f for f in _searchable_files(base) if rx.fullmatch(f.relative_to(base).as_posix())]
    if not matches:
        return "No files found"
    lines = [ws.relative(f) for f in matches[:_MAX_GLOB_RESULTS]]
    if len(matches) > _MAX_GLOB_RESULTS:
        lines.append(f"… {len(matches) - _MAX_GLOB_RESULTS} more (narrow the pattern)")
    return "\n".join(lines)


@tool(parse_docstring=True)
@_tool_errors
def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    ignore_case: bool = False,
    output_mode: Literal["content", "files", "count"] = "content",
    max_results: int = 100,
) -> str:
    """Search file contents with a regular expression (Python syntax). Files ignored by
    .gitignore and binary files are skipped.

    Args:
        pattern: Regular expression, e.g. "def \\w+_handler" or "TODO".
        path: File or directory to search, relative to the workspace root.
        glob: Only search files matching this glob; a pattern without "/" matches file names at any depth (e.g. "*.py").
        ignore_case: Case-insensitive search.
        output_mode: "content" (path:line: text), "files" (matching paths) or "count" (matches per file).
        max_results: Stop after this many matching lines (content) or files (files/count).
    """
    ws = get_workspace()
    base = ws.resolve(path, "read")
    if not base.exists():
        return f"Error: path not found: {ws.relative(base)}"
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"Error: invalid regular expression: {e}"
    file_rx = None
    if glob:
        file_rx = compile_glob(glob if "/" in glob else "**/" + glob)
    max_results = max(1, min(max_results, 1000))
    deadline = time.monotonic() + _GREP_DEADLINE_S

    out: list[str] = []
    hits = 0
    chars = 0
    stopped = ""
    for f in _searchable_files(base):
        if time.monotonic() > deadline:
            stopped = f"search stopped after {_GREP_DEADLINE_S:.0f}s; narrow path or glob"
            break
        rel = ws.relative(f)
        if file_rx and not file_rx.fullmatch(f.relative_to(base).as_posix() if base.is_dir() else f.name):
            continue
        try:
            if f.stat().st_size > _max_bytes() or _is_binary(f):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file_hits = 0
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line[:_GREP_SCAN_CHARS]):
                file_hits += 1
                if output_mode == "content":
                    entry = f"{rel}:{n}: {_clip(line)}"
                    out.append(entry)
                    hits += 1
                    chars += len(entry) + 1
                    if hits >= max_results or chars > _MAX_OUTPUT_CHARS:
                        break
                elif output_mode == "files":
                    break
        if file_hits and output_mode != "content":
            out.append(rel if output_mode == "files" else f"{rel}: {file_hits}")
            hits += 1
        if hits >= max_results or chars > _MAX_OUTPUT_CHARS:
            stopped = f"stopped at {hits} results; narrow the search or raise max_results"
            break

    if not out:
        return "No matches found" + (f" ({stopped})" if stopped else "")
    if stopped:
        out.append(f"… {stopped}")
    return "\n".join(out)


READ_TOOLS = [read_file, list_directory, file_exists, glob, grep]
WRITE_TOOLS = [write_file, edit_file, append_file, delete_file]
