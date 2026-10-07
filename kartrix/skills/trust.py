"""Trust-on-first-use for skills found in a workspace (B5).

A repo can ship ``.kartrix/skills/``; their text reaches the system prompt and the agent's
instructions, so a cloned repo could use them to steer the agent. A skill is used only after
the user trusted it (``/skills trust <name>``), and the trust is pinned to a SHA-256 over all of
the skill's files: any change makes it untrusted again until re-approved.

Trust lives in the per-user data folder (never in the workspace, which the repo controls),
keyed by the workspace root.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from kartrix.paths import user_data_dir

_FILE = "trusted_skills.json"


def skill_digest(skill_dir: Path) -> str:
    """SHA-256 over every file's relative path and content (links are recorded, never followed)."""
    h = hashlib.sha256()
    for path in sorted(skill_dir.rglob("*"), key=lambda p: p.relative_to(skill_dir).as_posix()):
        rel = path.relative_to(skill_dir).as_posix().encode()
        if path.is_symlink():
            h.update(b"L\0" + rel + b"\0" + os.readlink(path).encode() + b"\0")
        elif path.is_file():
            data = path.read_bytes()
            h.update(b"F\0" + rel + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def _key(workspace_root: Path) -> str:
    return os.path.normcase(str(workspace_root.resolve()))


def _load() -> dict[str, dict[str, str]]:
    try:
        data = json.loads((user_data_dir() / _FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: dict[str, dict[str, str]]) -> None:
    path = user_data_dir() / _FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def trusted_digests(workspace_root: Path) -> dict[str, str]:
    """Skill name → digest the user trusted, for this workspace."""
    entry = _load().get(_key(workspace_root), {})
    return {k: v for k, v in entry.items() if isinstance(k, str) and isinstance(v, str)}


def trust(workspace_root: Path, name: str, digest: str) -> None:
    data = _load()
    data.setdefault(_key(workspace_root), {})[name] = digest
    _save(data)


def untrust(workspace_root: Path, name: str) -> None:
    data = _load()
    data.get(_key(workspace_root), {}).pop(name, None)
    _save(data)
