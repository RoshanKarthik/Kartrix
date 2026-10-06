"""Built-in knowledge of what a command does (B2): argv → :class:`Category`.

The category, not the program name, decides allow / ask / deny per permission mode
(see ``kartrix.security.command_policy``). Unknown programs are ``UNKNOWN`` and so always
need approval outside read-only mode. Wrappers (``uv run``, ``python -m``, ``npx`` with a
local binary) are unwrapped and the inner command is classified too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath


class Category(StrEnum):
    READ = "read"  # inspects only
    WRITE = "write"  # creates/changes files in the workspace (like the file tools)
    RUN = "run"  # runs project code: tests, builds, linters, scripts, dev servers
    INSTALL = "install"  # project-local package installs (network to allow-listed registries)
    GIT_WRITE = "git_write"  # changes git state, recoverable
    NETWORK = "network"  # talks to arbitrary hosts or downloads and runs code
    DESTRUCTIVE = "destructive"  # deletes/discards data, kills processes
    UNKNOWN = "unknown"
    DENY = "deny"  # never, in any mode


# How restrictive a category is across modes; wrappers keep the stricter of two.
_RANK = {
    Category.READ: 0,
    Category.WRITE: 1,
    Category.RUN: 2,
    Category.INSTALL: 2,
    Category.GIT_WRITE: 2,
    Category.NETWORK: 3,
    Category.DESTRUCTIVE: 3,
    Category.UNKNOWN: 3,
    Category.DENY: 4,
}


@dataclass(frozen=True)
class Classification:
    category: Category
    reason: str
    installer: str | None = None  # pip/uv/npm/pnpm/yarn/bun/poetry: registry rules apply

    def at_least(self, floor: Category, reason: str) -> Classification:
        if _RANK[self.category] >= _RANK[floor]:
            return self
        return Classification(floor, reason, self.installer)


def _c(category: Category, reason: str, installer: str | None = None) -> Classification:
    return Classification(category, reason, installer)


def program_name(token: str) -> str:
    """``C:\\x\\NPM.cmd`` → ``npm``; ``/usr/bin/python3`` → ``python3``."""
    name = PurePath(token.replace("\\", "/")).name.lower()
    for ext in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


HARD_DENY = {
    # privilege escalation
    "sudo", "su", "doas", "runas", "pkexec", "gsudo",
    # system state, services, boot
    "shutdown", "reboot", "halt", "poweroff", "systemctl", "service", "launchctl", "bcdedit", "csrutil", "nvram",
    # disks
    "fdisk", "sfdisk", "parted", "diskpart", "format", "dd", "mount", "umount", "vssadmin", "cipher",
    # persistent system configuration, scheduled jobs, firewall, accounts
    "reg", "regedit", "setx", "schtasks", "crontab", "at", "sc", "netsh", "iptables", "nft", "ufw", "firewall-cmd",
    "wmic", "chown", "chgrp", "passwd", "useradd", "usermod", "userdel", "net", "defaults", "spctl", "xattr",
    # Windows binaries commonly abused to download or run code
    "certutil", "bitsadmin", "mshta", "rundll32", "regsvr32", "cscript", "wscript", "osascript",
    "wsl",
}  # fmt: skip

SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "cmd", "powershell", "pwsh"}
_SCRIPT_SUFFIXES = (".sh", ".bash", ".zsh", ".ps1")

READ_PROGRAMS = {
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "which", "where", "whoami", "file", "stat", "du", "df", "tree",
    "sort", "uniq", "diff", "cmp", "basename", "dirname", "realpath", "readlink", "grep", "egrep", "fgrep", "rg",
    "find", "date", "uname", "hostname", "cut", "jq", "sha256sum", "sha1sum", "md5sum", "shasum", "xxd", "od",
    "hexdump", "nl", "true", "false", "printenv",
}  # fmt: skip

WRITE_PROGRAMS = {"mkdir", "touch", "cp", "copy", "ln", "chmod"}

RUN_PROGRAMS = {
    # Python
    "pytest", "py.test", "tox", "nox", "mypy", "ruff", "black", "isort", "flake8", "pylint", "pyright", "bandit",
    "coverage", "alembic", "uvicorn", "gunicorn", "hypercorn", "flask", "fastapi", "django-admin",
    # JavaScript / TypeScript
    "tsc", "tsx", "ts-node", "vite", "vitest", "jest", "mocha", "eslint", "prettier", "next", "webpack", "rollup",
    "esbuild", "parcel", "prisma", "drizzle-kit",
    "make", "sqlite3",
}  # fmt: skip

NETWORK_PROGRAMS = {
    "curl", "wget", "ssh", "scp", "sftp", "rsync", "ftp", "telnet", "nc", "ncat", "netcat", "socat", "gh", "http",
    "https", "aria2c", "docker", "podman",
}  # fmt: skip

DESTRUCTIVE_PROGRAMS = {"rm", "rmdir", "del", "erase", "rd", "unlink", "shred", "mv", "move", "truncate", "kill",
                        "pkill", "killall", "taskkill"}  # fmt: skip

_PYTHON = re.compile(r"^(python[0-9.]*|py|pypy[0-9.]*)$")
_VERSION_ARGS = (["--version"], ["-V"], ["-v"], ["version"], ["--help"], ["-h"], ["help"])
_DLX = {"npx", "bunx", "pnpx", "uvx"}
_NODE_PMS = {"npm", "pnpm", "yarn", "bun"}
_GIT_CONFIG_SAFE_KEYS = {"user.name", "user.email", "init.defaultbranch", "core.autocrlf", "pull.rebase"}


def classify(argv: list[str], cwd: Path, depth: int = 0) -> Classification:
    """Categorise ``argv`` (already tokenised, no shell syntax). ``cwd`` is where it runs."""
    if not argv:
        return _c(Category.DENY, "empty command")
    if depth > 4:
        return _c(Category.UNKNOWN, "too many nested wrappers")
    prog = program_name(argv[0])
    args = argv[1:]

    if prog in HARD_DENY or prog.startswith("mkfs"):
        return _c(Category.DENY, f"{prog} changes the system outside the project")
    if args in _VERSION_ARGS:
        return _c(Category.READ, "version/help")
    if prog in SHELLS:
        return _classify_shell(prog, args)
    if _PYTHON.match(prog):
        return _classify_python(args, cwd, depth)
    if prog == "git":
        return _classify_git(args)
    if prog in _NODE_PMS:
        return _classify_node_pm(prog, args, cwd, depth)
    if prog in ("pip", "pip3"):
        return _classify_pip(args)
    if prog == "uv":
        return _classify_uv(args, cwd, depth)
    if prog == "poetry":
        return _classify_poetry(args, cwd, depth)
    if prog in _DLX:
        return _classify_dlx(prog, args, cwd, depth)
    if prog == "node":
        if any(a in ("-e", "--eval", "-p", "--print") for a in args):
            return _c(Category.UNKNOWN, "inline code (node -e)")
        return _c(Category.RUN, "runs a Node.js script")
    if prog in ("pipx",) or (prog == "playwright" and args[:1] == ["install"]):
        return _c(Category.NETWORK, f"{prog} downloads software")
    if prog in READ_PROGRAMS:
        return _classify_read(prog, args)
    if prog in WRITE_PROGRAMS:
        return _c(Category.WRITE, f"{prog} changes files")
    if prog in DESTRUCTIVE_PROGRAMS:
        return _c(Category.DESTRUCTIVE, f"{prog} deletes, moves or kills")
    if prog in NETWORK_PROGRAMS:
        return _c(Category.NETWORK, f"{prog} talks to other hosts")
    if prog in RUN_PROGRAMS or prog == "playwright":
        return _c(Category.RUN, f"runs {prog}")
    return _c(Category.UNKNOWN, f"{prog} is not a known command")


def _classify_read(prog: str, args: list[str]) -> Classification:
    if prog == "find" and any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls") for a in args):
        return _c(Category.DESTRUCTIVE, "find with -delete/-exec")
    if prog == "rg" and any(a.startswith("--pre") for a in args):
        return _c(Category.UNKNOWN, "rg --pre runs another program")
    if prog == "sort" and any(a in ("-o", "--output") or a.startswith("--output=") for a in args):
        return _c(Category.WRITE, "sort -o writes a file")
    if prog == "uniq" and len([a for a in args if not a.startswith("-")]) > 1:
        return _c(Category.WRITE, "uniq writes its output file")
    return _c(Category.READ, f"{prog} only reads")


def _classify_shell(prog: str, args: list[str]) -> Classification:
    script = next((a for a in args if not a.startswith(("-", "/"))), None)
    inline = any(
        a.lower() in ("-c", "/c", "/k", "/r") or (prog in ("powershell", "pwsh") and a.lower().startswith(("-c", "-e")))
        for a in args
    )
    if inline or prog == "cmd":
        return _c(Category.DENY, "inline shell code bypasses the command policy — run the program directly")
    if script and script.lower().endswith(_SCRIPT_SUFFIXES):
        return _c(Category.RUN, f"runs the script {script}")
    return _c(Category.DENY, f"interactive {prog} shells are not allowed — run the program directly")


_PY_OPTS_WITH_VALUE = {"-X", "-W", "-Q"}


def _classify_python(args: list[str], cwd: Path, depth: int) -> Classification:
    i = 0
    while i < len(args) and args[i].startswith("-") and args[i] not in ("-c", "-m"):
        i += 2 if args[i] in _PY_OPTS_WITH_VALUE else 1
    rest = args[i:]
    if not rest:
        return _c(Category.RUN, "python interpreter")
    if rest[0] == "-c":
        return _c(Category.UNKNOWN, "inline code (python -c)")
    if rest[0] == "-m":
        if len(rest) < 2:
            return _c(Category.DENY, "python -m without a module")
        module, margs = rest[1], rest[2:]
        if module in ("pip", "pip3"):
            return _classify_pip(margs)
        if module in ("venv", "virtualenv", "unittest", "http.server", "compileall", "build"):
            return _c(Category.RUN, f"python -m {module}")
        if module in ("json.tool", "pydoc", "site", "platform", "sysconfig", "tabnanny"):
            return _c(Category.READ, f"python -m {module}")
        inner = classify([module, *margs], cwd, depth + 1)
        return inner.at_least(Category.RUN, f"python -m {module}")
    return _c(Category.RUN, f"runs the Python script {rest[0]}")


def _git_subcommand(args: list[str]) -> tuple[str | None, list[str], Classification | None]:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a == "-c" or a.startswith(("--config-env", "--exec-path")):
            return None, [], _c(Category.DENY, "git -c/--config-env can run arbitrary programs (pager, ssh, aliases)")
        i += 2 if a in ("-C", "--git-dir", "--work-tree", "--namespace") else 1
    if i >= len(args):
        return None, [], _c(Category.READ, "git without a subcommand")
    return args[i], args[i + 1 :], None


_GIT_READ = {
    "status", "diff", "log", "show", "blame", "ls-files", "ls-tree", "rev-parse", "describe", "shortlog", "grep",
    "reflog", "cat-file", "show-ref", "count-objects", "rev-list", "merge-base", "check-ignore", "name-rev", "version",
    "help", "whatchanged", "show-branch", "var",
}  # fmt: skip
_GIT_WRITE = {"add", "commit", "init", "merge", "rebase", "cherry-pick", "revert", "mv", "apply", "am", "notes",
              "worktree", "bisect", "switch"}  # fmt: skip
_GIT_NETWORK = {"clone", "fetch", "pull", "ls-remote", "submodule", "remote-https", "archive", "send-email"}
_GIT_DESTRUCTIVE = {"clean", "gc", "prune", "filter-branch", "filter-repo", "update-ref", "replace"}


def _discard(sub: str) -> Classification:
    return _c(Category.DESTRUCTIVE, f"git {sub} can discard uncommitted changes")


def _classify_git(args: list[str]) -> Classification:
    sub, rest, early = _git_subcommand(args)
    if early is not None:
        return early
    if sub is None:
        return _c(Category.READ, "git without a subcommand")
    flags = set(rest)
    positional = [a for a in rest if not a.startswith("-")]

    if sub in _GIT_READ:
        return _c(Category.READ, f"git {sub} only reads")
    if sub == "config":
        if flags & {"--global", "--system", "--file", "-f", "--blob"}:
            return _c(Category.DENY, "git config outside this repository")
        if flags & {"--get", "--get-all", "--get-regexp", "--list", "-l"} or positional[:1] in (["get"], ["list"]):
            return _c(Category.READ, "git config read")
        key = (positional[1] if positional[:1] == ["set"] and len(positional) > 1 else positional[:1] or [""])[0]
        if key.lower() in _GIT_CONFIG_SAFE_KEYS:
            return _c(Category.GIT_WRITE, f"git config {key}")
        return _c(Category.DENY, "git config can make git run arbitrary programs (hooksPath, pager, aliases)")
    if sub == "branch":
        if flags & {"-D", "-d", "--delete", "-M", "--force", "-f"}:
            return _c(Category.DESTRUCTIVE, "git branch delete/force")
        return _c(Category.GIT_WRITE if positional else Category.READ, "git branch")
    if sub == "tag":
        if flags & {"-d", "--delete", "-f", "--force"}:
            return _c(Category.DESTRUCTIVE, "git tag delete/force")
        return _c(Category.GIT_WRITE if positional and "-l" not in flags else Category.READ, "git tag")
    if sub == "remote":
        return _c(
            Category.READ if not positional or positional[0] in ("show", "get-url") else Category.GIT_WRITE,
            "git remote",
        )
    if sub == "stash":
        action = positional[0] if positional else "push"
        if action in ("list", "show"):
            return _c(Category.READ, "git stash list/show")
        if action in ("drop", "clear"):
            return _c(Category.DESTRUCTIVE, "git stash drop/clear")
        return _c(Category.GIT_WRITE, f"git stash {action}")
    if sub == "reset":
        if flags & {"--hard", "--merge", "--keep"}:
            return _c(Category.DESTRUCTIVE, "git reset --hard discards changes")
        return _c(Category.GIT_WRITE, "git reset")
    if sub == "restore":
        staged_only = flags & {"--staged", "-S"} and not flags & {"--worktree", "-W"}
        return _c(Category.GIT_WRITE, "git restore --staged") if staged_only else _discard(sub)
    if sub == "checkout":
        if "--" in flags or "." in positional or flags & {"-f", "--force"}:
            return _discard(sub)
        return _c(Category.GIT_WRITE, "git checkout")
    if sub == "switch" and flags & {"-f", "--force", "--discard-changes"}:
        return _c(Category.DESTRUCTIVE, "git switch --discard-changes")
    if sub == "rm":
        return _c(Category.GIT_WRITE if "--cached" in flags else Category.DESTRUCTIVE, "git rm")
    if sub == "push":
        forced = flags & {"-f", "--force", "--force-with-lease", "--delete", "-d", "--mirror", "--prune"}
        if forced or any(p.startswith(("+", ":")) for p in positional):
            return _c(Category.DESTRUCTIVE, "git push that rewrites or deletes remote history")
        return _c(Category.NETWORK, "git push publishes to a remote")
    if sub in _GIT_WRITE:
        return _c(Category.GIT_WRITE, f"git {sub}")
    if sub in _GIT_NETWORK:
        return _c(Category.NETWORK, f"git {sub} talks to a remote")
    if sub in _GIT_DESTRUCTIVE:
        return _c(Category.DESTRUCTIVE, f"git {sub} can lose data")
    return _c(Category.UNKNOWN, f"git {sub} is not a known git command")


def _first_positional(args: list[str]) -> tuple[str | None, list[str]]:
    for i, a in enumerate(args):
        if not a.startswith("-"):
            return a, args[i + 1 :]
    return None, []


def _package_scripts(cwd: Path) -> set[str]:
    try:
        data = json.loads((cwd / "package.json").read_text(encoding="utf-8"))
        scripts = data.get("scripts", {})
        return set(scripts) if isinstance(scripts, dict) else set()
    except (OSError, ValueError, AttributeError):
        return set()


_GLOBAL_FLAGS = {"-g", "--global", "--location=global"}
_NPM_INSTALL = {"install", "i", "ci", "add", "uninstall", "remove", "rm", "un", "r", "update", "up", "upgrade",
                "dedupe", "prune", "isntall", "in"}  # fmt: skip
_NPM_RUN = {"run", "run-script", "rum", "urn", "test", "t", "tst", "start", "stop", "restart", "rebuild"}
_NPM_READ = {"ls", "list", "ll", "la", "outdated", "view", "info", "show", "v", "why", "explain", "root", "prefix",
             "bin", "help", "audit", "doctor", "fund", "search", "docs", "query", "version", "config", "get"}  # fmt: skip
_NPM_DENY = {"publish", "unpublish", "deprecate", "owner", "access", "adduser", "login", "logout", "token", "team",
             "dist-tag", "link", "ln", "set", "star", "unstar", "hook", "org", "profile"}  # fmt: skip


def _classify_node_pm(prog: str, args: list[str], cwd: Path, depth: int) -> Classification:
    if _GLOBAL_FLAGS & set(args) or any(a == "--location" for a in args):
        return _c(Category.DENY, "global installs change the user's machine — install into the project instead")
    sub, rest = _first_positional(args)
    if sub is None:
        if prog in ("yarn", "bun"):
            return _c(Category.INSTALL, f"{prog} install", prog)
        return _c(Category.UNKNOWN, f"{prog} without a subcommand")
    if sub in _NPM_DENY:
        return _c(Category.DENY, f"{prog} {sub} publishes or changes user/registry settings — do it yourself")
    if sub == "config" and rest[:1] not in (["get"], ["list"], ["ls"]):
        return _c(Category.DENY, f"{prog} config changes user settings")
    if sub in ("pkg",) and rest[:1] == ["get"]:
        return _c(Category.READ, f"{prog} pkg get")
    if sub == "audit" and "fix" in rest:
        return _c(Category.INSTALL, f"{prog} audit fix", prog)
    if sub in _NPM_INSTALL:
        return _c(Category.INSTALL, f"{prog} {sub}", prog)
    if sub in _NPM_RUN or sub == "pkg":
        return _c(Category.RUN, f"{prog} {sub} runs project scripts")
    if sub in ("exec", "x", "dlx"):
        return _classify_dlx(prog, rest, cwd, depth)
    if sub in ("create", "init") and (sub == "create" or [a for a in rest if not a.startswith("-")]):
        return _c(Category.NETWORK, f"{prog} {sub} downloads and runs a project generator")
    if sub == "init":
        return _c(Category.WRITE, f"{prog} init writes package.json")
    if sub in _NPM_READ:
        return _c(Category.READ, f"{prog} {sub}")
    if prog != "npm" and sub in _package_scripts(cwd):
        return _c(Category.RUN, f"{prog} {sub} runs a package.json script")
    return _c(Category.UNKNOWN, f"{prog} {sub} is not a known command")


def _classify_pip(args: list[str]) -> Classification:
    sub, rest = _first_positional(args)
    flags = set(rest)
    if sub in ("install", "uninstall", "wheel"):
        if flags & {"--user", "--break-system-packages", "--root"} or any(a.startswith("--root=") for a in rest):
            return _c(Category.DENY, "installs outside the project environment (--user/--root/--break-system-packages)")
        return _c(Category.INSTALL, f"pip {sub}", "pip")
    if sub == "download":
        return _c(Category.NETWORK, "pip download")
    if sub in ("list", "show", "freeze", "check", "inspect", "help", "index", "debug", "hash"):
        return _c(Category.READ, f"pip {sub}")
    if sub == "config":
        if rest[:1] in (["list"], ["get"], ["debug"]):
            return _c(Category.READ, "pip config read")
        return _c(Category.DENY, "pip config changes user settings")
    return _c(Category.UNKNOWN, f"pip {sub} is not a known command")


_UV_OPTS_WITH_VALUE = {
    "--with", "--with-editable", "--with-requirements", "--python", "-p", "--package", "--directory", "--project",
    "--env-file", "--extra", "--group", "--only-group", "--no-group", "--index", "--default-index", "--index-url",
    "--extra-index-url", "-i", "--find-links", "-f", "--config-file", "--cache-dir", "--script",
}  # fmt: skip


def _split_wrapper(args: list[str], opts_with_value: set[str]) -> tuple[list[str], list[str]]:
    """Split ``[opts..., cmd, args...]`` into options and the wrapped command."""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return args[:i], args[i + 1 :]
        if not a.startswith("-"):
            break
        i += 2 if a in opts_with_value else 1
    return args[:i], args[i:]


def _classify_uv(args: list[str], cwd: Path, depth: int) -> Classification:
    sub, rest = _first_positional(args)
    if sub in ("add", "remove", "sync", "lock"):
        return _c(Category.INSTALL, f"uv {sub}", "uv")
    if sub == "pip":
        action = rest[0] if rest else ""
        if "--system" in rest or "--break-system-packages" in rest:
            return _c(Category.DENY, "installs into the system Python")
        if action in ("install", "uninstall", "sync", "compile"):
            return _c(Category.INSTALL, f"uv pip {action}", "uv")
        if action in ("list", "show", "freeze", "tree", "check"):
            return _c(Category.READ, f"uv pip {action}")
        return _c(Category.UNKNOWN, f"uv pip {action} is not a known command")
    if sub == "run":
        opts, inner = _split_wrapper(rest, _UV_OPTS_WITH_VALUE)
        if not inner:
            return _c(Category.UNKNOWN, "uv run without a command")
        inner_cls = classify(inner, cwd, depth + 1)
        if any(o.startswith("--with") for o in opts) and _RANK[inner_cls.category] <= _RANK[Category.INSTALL]:
            return _c(Category.INSTALL, "uv run --with installs packages", "uv")  # registry rules apply
        return inner_cls.at_least(Category.RUN, "uv run")
    if sub in ("tree", "export", "version"):
        return _c(Category.READ, f"uv {sub}")
    if sub in ("venv", "init", "build"):
        return _c(Category.RUN, f"uv {sub}")
    if sub in ("tool", "python", "self", "publish"):
        if sub == "tool" and rest[:1] == ["run"]:
            return _classify_dlx("uvx", rest[1:], cwd, depth)
        if sub == "python" and rest[:1] in (["list"], ["find"], ["dir"]):
            return _c(Category.READ, f"uv python {rest[0]}")
        return _c(Category.DENY, f"uv {sub} changes the user's machine or publishes — do it yourself")
    return _c(Category.UNKNOWN, f"uv {sub} is not a known command")


def _classify_poetry(args: list[str], cwd: Path, depth: int) -> Classification:
    sub, rest = _first_positional(args)
    if sub in ("add", "remove", "install", "update", "lock", "sync"):
        return _c(Category.INSTALL, f"poetry {sub}", "poetry")
    if sub == "run":
        return classify(rest, cwd, depth + 1).at_least(Category.RUN, "poetry run") if rest else _c(Category.UNKNOWN, "")
    if sub in ("show", "check", "env", "version", "about"):
        return _c(Category.READ, f"poetry {sub}")
    if sub in ("new", "init", "build"):
        return _c(Category.RUN, f"poetry {sub}")
    if sub in ("publish", "config", "self", "source"):
        return _c(Category.DENY, f"poetry {sub} publishes or changes user settings — do it yourself")
    return _c(Category.UNKNOWN, f"poetry {sub} is not a known command")


def _local_bin(name: str, cwd: Path) -> bool:
    for d in (cwd, *cwd.parents):
        bin_dir = d / "node_modules" / ".bin"
        if any((bin_dir / (name + ext)).is_file() for ext in ("", ".cmd", ".exe")):
            return True
        if (d / ".git").exists():  # don't look above the repository
            break
    return False


def _classify_dlx(prog: str, args: list[str], cwd: Path, depth: int) -> Classification:
    opts, inner = _split_wrapper(args, {"-p", "--package", "-c", "--call", "--from", "--with", "--python"})
    if any(o in ("-c", "--call") for o in opts):
        return _c(Category.DENY, f"{prog} -c runs a shell string — run the program directly")
    if not inner:
        return _c(Category.UNKNOWN, f"{prog} without a command")
    name = inner[0]
    if prog not in ("uvx",) and not any(o in ("-p", "--package") for o in opts) and _local_bin(name, cwd):
        return classify(inner, cwd, depth + 1).at_least(Category.RUN, f"{prog} {name} (installed in node_modules)")
    return _c(Category.NETWORK, f"{prog} downloads and runs {name} from the registry")
