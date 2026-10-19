"""``constraints.txt`` (what installs use) must match ``uv.lock`` (what the tests ran with).

Regenerate after changing the lock:
``uv export --frozen --no-default-groups --no-hashes --no-emit-project --no-annotate -o constraints.txt``
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _locked_runtime_versions() -> dict[str, set[str]]:
    """Every package Kartrix needs at run time (no dev or eval groups), with its locked versions."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    by_name: dict[str, list[dict[str, Any]]] = {}
    for package in lock["package"]:
        by_name.setdefault(package["name"], []).append(package)
    needed: dict[str, set[str]] = {}
    todo: list[tuple[str, str | None, list[str]]] = [("kartrix", None, [])]
    seen: set[tuple[str, str | None, tuple[str, ...]]] = set()
    while todo:
        name, version, extras = todo.pop()
        key = (name, version, tuple(extras))
        if key in seen:
            continue
        seen.add(key)
        for package in by_name[name]:
            if version is not None and package["version"] != version:
                continue
            if name != "kartrix":
                needed.setdefault(name, set()).add(package["version"])
            deps = list(package.get("dependencies", []))
            for extra in extras:
                deps += package.get("optional-dependencies", {}).get(extra, [])
            todo += [(d["name"], d.get("version"), list(d.get("extra", []))) for d in deps]
    return needed


def _constraints() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split(";")[0].strip()
        if line and not line.startswith("#"):
            match = re.fullmatch(r"([A-Za-z0-9._-]+)==(\S+)", line)
            assert match, f"not an exact pin: {line!r}"
            pins[match[1].lower()] = match[2]
    return pins


def test_constraints_pin_exactly_the_locked_runtime_packages() -> None:
    locked, pinned = _locked_runtime_versions(), _constraints()
    assert set(pinned) == set(locked), "constraints.txt is out of date: regenerate it (see this module's docstring)"
    stale = {name: (version, locked[name]) for name, version in pinned.items() if version not in locked[name]}
    assert not stale, f"constraints.txt is out of date (pinned, locked): {stale}"
