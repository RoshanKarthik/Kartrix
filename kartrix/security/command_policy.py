"""Command policy engine (B2): decide allow / ask / deny before anything runs.

A command string from the model goes through:

1. **Parse** with ``shlex`` (POSIX rules; backslashes are literal on Windows). Shell syntax
   — pipes, ``&&``, ``;``, redirects, ``$(...)``, backticks — is refused: commands run with
   ``shell=False``, one program per call.
2. **Classify** (``command_rules``): what the command does → a :class:`Category`.
3. **Resolve the executable** from ``PATH`` only — never the workspace root or a relative
   ``PATH`` entry, so a planted ``git.bat`` can't hijack ``git``. A program given by path
   inside the workspace counts as running project code.
4. **Check arguments that look like paths** against the workspace jail, so ``cat .env`` or
   ``rm ../x`` can't sidestep the file tools' rules.
5. **Check package installs**: index/registry options, requirements files and ``.npmrc`` may
   only point at ``permissions.registries``; direct URL installs need approval.
6. **Windows batch files** (``npm.cmd`` …) get a quoted command line; arguments cmd.exe
   would still interpret (``%``, ``"``) are refused.
7. **User rules** (``permissions.deny`` / ``ask`` / ``allow``), then the **mode** table, adjusted
   for the sandbox (B8, :mod:`kartrix.sandbox`): with one, ``default`` mode also runs project code
   and registry installs (where the sandbox limits the network to the registries) without asking;
   without one, ``auto`` mode asks before running project code or installing.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from kartrix.config import settings
from kartrix.observability.logger import get_logger
from kartrix.sandbox.base import Backend, Network
from kartrix.sandbox.manager import argv_unsandboxed, network_for, require_native, sandbox_for
from kartrix.security.command_rules import Category, Classification, classify, program_name
from kartrix.security.permissions import Mode, get_mode
from kartrix.security.workspace import Workspace, WorkspaceError, get_workspace

logger = get_logger(__name__)

Action = Literal["allow", "ask", "deny"]

_WINDOWS = os.name == "nt"
_ALLOW: Action = "allow"
_ASK: Action = "ask"
_DENY: Action = "deny"

MODE_ACTIONS: dict[str, dict[Category, Action]] = {
    "read_only": {c: (_ALLOW if c is Category.READ else _DENY) for c in Category},
    "default": {
        **{c: _ASK for c in Category},
        Category.READ: _ALLOW,
        Category.WRITE: _ALLOW,
        Category.DENY: _DENY,
    },
    "auto": {
        **{c: _ALLOW for c in Category},
        Category.NETWORK: _ASK,
        Category.DESTRUCTIVE: _ASK,
        Category.UNKNOWN: _ASK,
        Category.DENY: _DENY,
    },
}


def mode_action(mode: Mode, category: Category, sandbox: Backend | None) -> Action:
    """The mode table, adjusted for whether (and how well) the command will be sandboxed."""
    action = MODE_ACTIONS[mode][category]
    if mode == "default" and sandbox is not None:
        if category is Category.RUN or (category is Category.INSTALL and sandbox.registries_enforced):
            return _ALLOW
    if mode == "auto" and sandbox is None and category in (Category.RUN, Category.INSTALL):
        return _ASK
    return action


@dataclass(frozen=True)
class Decision:
    action: Action
    category: Category
    reason: str
    command: str
    argv: list[str] = field(default_factory=list)
    cwd: Path | None = None
    executable: Path | None = None
    cmdline: str | None = None  # Windows batch targets: the pre-quoted command line to run
    network: Network | None = None  # set when the command runs sandboxed

    @property
    def run_args(self) -> list[str] | str:
        if self.cmdline is not None:
            return self.cmdline
        assert self.executable is not None  # noqa: S101 — only allowed decisions are run
        return [str(self.executable), *self.argv[1:]]


class _Deny(Exception):
    def __init__(self, reason: str, category: Category = Category.DENY) -> None:
        super().__init__(reason)
        self.category = category


# ── parsing ───────────────────────────────────────────────────────────

_OPERATOR = re.compile(r"^[();<>|&]+$")


def parse_command(command: str) -> list[str]:
    """Tokenise like a POSIX shell would, but refuse anything only a shell could run."""
    if not command or not command.strip():
        raise _Deny("command cannot be empty")
    if "\x00" in command or "\n" in command or "\r" in command:
        raise _Deny("run one command per call (no newlines or NUL bytes)")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    if _WINDOWS:
        lexer.escape = ""  # C:\path\to — backslash is a path separator, not an escape
    try:
        argv = list(lexer)
    except ValueError as e:
        raise _Deny(f"could not parse the command ({e})") from None
    for tok in argv:
        if _OPERATOR.match(tok):
            raise _Deny(
                f"shell syntax {tok!r} is not supported — commands run without a shell, one program per call. "
                "Use the directory argument instead of cd, and grep/read_file instead of pipes"
            )
        if "$(" in tok or "`" in tok:
            raise _Deny("command substitution ($(...) or backticks) is not supported — commands run without a shell")
    if not argv:
        raise _Deny("command cannot be empty")
    return argv


# ── executable resolution ─────────────────────────────────────────────

_WIN_EXTS = (".com", ".exe", ".bat", ".cmd")  # not PATHEXT: .js/.vbs/.wsf would run via Windows Script Host


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.realpath(p))


def find_executable(name: str, path_env: str, workspace_root: Path) -> Path | None:
    """Look ``name`` up on ``PATH``, skipping relative entries and the workspace root."""
    root = _norm(workspace_root)
    has_ext = name.lower().endswith(_WIN_EXTS)
    for d in path_env.split(os.pathsep):
        d = d.strip().strip('"')
        if not d or not os.path.isabs(d) or _norm(d) == root:
            continue
        candidates = [name] if not _WINDOWS or has_ext else [name + ext for ext in _WIN_EXTS]
        for c in candidates:
            p = Path(d) / c
            if p.is_file() and (_WINDOWS or os.access(p, os.X_OK)):
                return p
    return None


def _resolve_executable(argv0: str, cwd: Path, ws: Workspace) -> tuple[Path, bool]:
    """Absolute executable path, and whether it lives inside the workspace."""
    path_env = os.environ.get("PATH", "")
    if "/" not in argv0 and "\\" not in argv0:
        exe = find_executable(argv0, path_env, ws.root)
        if exe is None:
            hint = " (cmd built-ins like dir, type, copy aren't programs — use the file tools)" if _WINDOWS else ""
            raise _Deny(f"command not found: {argv0}{hint}", Category.UNKNOWN)
        return exe, _norm(exe).startswith(_norm(ws.root) + os.sep)

    real = Path(os.path.realpath(cwd / argv0))
    if not _norm(real).startswith(_norm(ws.root) + os.sep):
        path_dirs = {_norm(d) for d in path_env.split(os.pathsep) if d and os.path.isabs(d)}
        if _norm(real.parent) not in path_dirs:
            raise _Deny(f"{argv0} is outside the workspace and not in a PATH directory")
        if not real.is_file():
            raise _Deny(f"command not found: {argv0}", Category.UNKNOWN)
        return real, False
    try:
        target = ws.resolve(str(cwd / argv0), "read")
    except WorkspaceError as e:
        raise _Deny(str(e)) from None
    if _WINDOWS and not target.suffix and not target.is_file():
        target = next((target.with_suffix(e) for e in _WIN_EXTS if target.with_suffix(e).is_file()), target)
    if not target.is_file():
        raise _Deny(f"command not found: {argv0}", Category.UNKNOWN)
    return target, True


_BATCH_UNSAFE = re.compile(r'["%\r\n]')
_BATCH_NEEDS_QUOTES = re.compile(r"[\s&|<>^(),;=!]")


def batch_command_line(exe: Path, args: list[str]) -> str:
    """Command line for a .bat/.cmd target that cmd.exe can't reinterpret (BatBadBut)."""
    parts = [f'"{exe}"']
    for a in args:
        if _BATCH_UNSAFE.search(a) or a.endswith("\\"):
            raise _Deny(f"argument {a!r} can't be passed safely to the batch file {exe.name} (contains % or \")")
        parts.append(f'"{a}"' if not a or _BATCH_NEEDS_QUOTES.search(a) else a)
    return " ".join(parts)


# ── argument checks ───────────────────────────────────────────────────

_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_DEVICES = {"/dev/null", "nul", "-"}


def _path_values(argv: list[str]) -> Iterator[str]:
    for tok in argv[1:]:
        value = tok.split("=", 1)[1] if tok.startswith("-") and "=" in tok else tok
        if tok.startswith("-") and "=" not in tok:
            continue
        value = value.lstrip("@")  # curl -d @file, npm @scope/pkg
        if not value or _URL.match(value) or value.startswith(("git+", "npm:")) or value.lower() in _DEVICES:
            continue
        # "rev:path" (git), "host:path" (scp), "src:dst" (docker -v): check each side.
        if ":" in value and not _DRIVE.match(value):
            yield from (p for p in value.split(":") if p)
        else:
            yield value


def _looks_like_path(value: str, cwd: Path, strict: bool) -> bool:
    """Heuristic: is this argument meant as a file path? ``strict`` (commands that write or
    delete) treats every absolute-looking value as a path; otherwise an absolute value counts
    only if it or its (non-root) parent exists, so a pattern like "/api" isn't a path."""
    try:
        if (cwd / value).exists():
            return True
        if any(c.isspace() for c in value):
            return False  # free text such as a commit message
        if value.startswith(("/", "\\")) or _DRIVE.match(value):
            p = Path(value)
            return strict or p.exists() or (p.parent != Path(p.anchor) and p.parent.exists())
    except (OSError, ValueError):
        return False
    return value.startswith(("~", ".")) or "/" in value or "\\" in value


def _check_paths(argv: list[str], cwd: Path, ws: Workspace, category: Category) -> None:
    strict = category in (Category.WRITE, Category.DESTRUCTIVE)
    for value in _path_values(argv):
        if not _looks_like_path(value, cwd, strict):
            continue
        if value.startswith("~"):
            raise _Deny(f"{value}: paths in the home directory are outside the workspace")
        try:
            ws.resolve(value if os.path.isabs(value) else str(cwd / value), "read")
        except WorkspaceError as e:
            raise _Deny(f"argument {e}") from None
        except (OSError, ValueError):
            continue  # not a usable path (e.g. a glob with characters the OS rejects)


# ── package installs ──────────────────────────────────────────────────

_INDEX_OPTS = {
    "--index-url", "-i", "--extra-index-url", "--find-links", "-f", "--trusted-host", "--registry",
    "--default-index", "--index",
}  # fmt: skip
_URL_ANYWHERE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_REQ_OPTS = {"-r", "--requirement", "-c", "--constraint", "--with-requirements"}
_NODE_INSTALLERS = {"npm", "pnpm", "yarn", "bun"}


def _host_allowed(value: str) -> bool:
    url = value.split("=", 1)[1] if "=" in value and not _URL.match(value) else value  # uv: --index name=url
    host = ((urlsplit(url).hostname or "") if _URL.match(url) else url.split("/")[0]).lower().strip()
    allowed = [r.lower() for r in settings.permissions.registries]
    return any(host == r or host.endswith("." + r) for r in allowed)


def _option_values(tokens: list[str], names: set[str]) -> Iterator[tuple[str, str]]:
    for i, tok in enumerate(tokens):
        if tok.startswith("--") and "=" in tok:
            name, value = tok.split("=", 1)
            if name in names:
                yield name, value
        elif tok in names and i + 1 < len(tokens):
            yield tok, tokens[i + 1]


def _check_index(name: str, value: str, source: str) -> None:
    looks_remote = _URL.match(value) or name == "--trusted-host" or name in ("--registry", "--index", "--default-index")
    if looks_remote and not _host_allowed(value):
        raise _Deny(
            f"{source}: {name} {value} is not an allowed package registry "
            f"(allowed: {', '.join(settings.permissions.registries)})"
        )


def _check_install(argv: list[str], cls: Classification, cwd: Path, ws: Workspace) -> Category:
    """Deny non-allow-listed registries; return NETWORK for direct-URL installs, else the category."""
    tokens = argv[1:]
    for name, value in _option_values(tokens, _INDEX_OPTS):
        _check_index(name, value, "command")

    for _, req in _option_values(tokens, _REQ_OPTS):
        try:
            req_path = ws.resolve(req if os.path.isabs(req) else str(cwd / req), "read")
            lines = req_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (WorkspaceError, OSError) as e:
            raise _Deny(f"can't check requirements file {req}: {e}") from None
        for line in lines:
            parts = line.split("#", 1)[0].split()
            for name, value in _option_values(parts, _INDEX_OPTS):
                _check_index(name, value, req)
            if parts and (_URL.match(parts[0]) or parts[0].startswith("git+")) and not _host_allowed(parts[0]):
                return Category.NETWORK

    if cls.installer in _NODE_INSTALLERS:
        for npmrc in {cwd / ".npmrc", ws.root / ".npmrc"}:
            try:
                lines = npmrc.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                key, _, value = line.partition("=")
                if key.strip().endswith("registry") and value.strip() and not _host_allowed(value.strip()):
                    raise _Deny(f".npmrc sets registry {value.strip()}, which is not an allowed package registry")

    for tok in tokens:
        url = _URL_ANYWHERE.search(tok)
        if (url and not _host_allowed(url.group())) or tok.startswith(("github:", "gitlab:", "bitbucket:")):
            return Category.NETWORK  # direct URL / VCS install: not via an allowed registry
    return cls.category


# ── user rules ────────────────────────────────────────────────────────


def _rule_matches(rule: str, argv: list[str]) -> bool:
    try:
        tokens = shlex.split(rule, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    target = [program_name(argv[0]), *argv[1:]]
    tokens[0] = program_name(tokens[0])
    for i, tok in enumerate(tokens):
        if tok == "*" and i == len(tokens) - 1:
            return True
        if i >= len(target) or (tok != "*" and tok != target[i]):
            return False
    return len(target) == len(tokens)


def _user_rule(argv: list[str]) -> Action | None:
    cfg = settings.permissions
    for action, rules in ((_DENY, cfg.deny), (_ASK, cfg.ask), (_ALLOW, cfg.allow)):
        if any(_rule_matches(r, argv) for r in rules):
            return action
    return None


# ── entry point ───────────────────────────────────────────────────────


def evaluate(command: str, directory: str = ".", mode: Mode | None = None, *, log: bool = True) -> Decision:
    """Decide what to do with ``command`` run in ``directory`` (relative to the workspace).

    Denials are logged at WARNING, other decisions at INFO ("ask" is a normal path now that the
    user is asked); ``log=False`` for checks that don't act on the decision (approval pre-check)."""
    mode = mode or get_mode()
    ws = get_workspace()
    argv: list[str] = []
    cwd: Path | None = None
    try:
        try:
            cwd = ws.resolve(directory, "read")
        except WorkspaceError as e:
            raise _Deny(f"directory {e}") from None
        if not cwd.is_dir():
            raise _Deny(f"not a directory: {directory}")
        argv = parse_command(command)
        cls = classify(argv, cwd)
        if cls.category is Category.DENY:
            raise _Deny(cls.reason)
        exe, inside = _resolve_executable(argv[0], cwd, ws)
        if inside:  # venv / node_modules binaries, scripts: project code, never "read"
            cls = cls.at_least(Category.RUN, f"runs {exe.name} from inside the workspace")
        category = cls.category
        _check_paths(argv, cwd, ws, category)
        if cls.installer is not None or category is Category.INSTALL:
            category = _check_install(argv, cls, cwd, ws)
        cmdline = batch_command_line(exe, argv[1:]) if _WINDOWS and exe.suffix.lower() in (".bat", ".cmd") else None

        sandbox = sandbox_for(argv)
        exempt = sandbox is None and not argv_unsandboxed(argv)  # would be sandboxed, but none exists
        if exempt and require_native():
            raise _Deny("sandbox.backend is 'native' but no sandbox is available here (run `kartrix sandbox`)")
        rule = _user_rule(argv)
        action: Action = mode_action(mode, category, sandbox)
        reason = cls.reason
        if rule is not None and mode != "read_only":
            action, reason = rule, f"{reason}; matched a permissions.{rule} rule"
        elif rule == _DENY:
            action = _DENY
        network = network_for(category, sandbox) if sandbox is not None else None
        if sandbox is not None and network is not None:
            reason = f"{reason} ({sandbox.label(network)})"
        elif exempt and category not in (Category.READ, Category.WRITE):
            reason = f"{reason} (not sandboxed: no sandbox available)"
        decision = Decision(action, category, reason, command, argv, cwd, exe, cmdline, network)
    except _Deny as e:
        decision = Decision(_DENY, e.category, str(e), command, argv, cwd)

    if not log:
        return decision
    emit = logger.warning if decision.action == _DENY else logger.info
    emit(
        "Command policy decision",
        extra={
            "action": decision.action,
            "category": str(decision.category),
            "reason": decision.reason,
            "argv": decision.argv or command,
            "mode": mode,
        },
    )
    return decision
