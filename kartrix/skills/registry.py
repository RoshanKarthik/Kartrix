from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kartrix.observability.logger import get_logger
from kartrix.security.injection import Finding, scan, strip_hidden, visible
from kartrix.skills import trust

logger = get_logger(__name__)

# Every skill package must have this file at its root.
# Folders inside skills_dir that lack this file are skipped entirely.
SKILL_FILENAME = "SKILL.md"

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_META_CHARS = 200  # description / when_to_use reach the system prompt: keep them short

Status = Literal["trusted", "untrusted", "changed"]


class SkillNotFoundError(Exception):
    pass


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str
    body: str
    skill_dir: Path
    digest: str
    findings: list[Finding] = field(default_factory=list)  # injection heuristics on the whole skill


class SkillRegistry:
    """
    Catalog of the skills found in skills_dir (normally ``<workspace>/.kartrix/skills``).

    Expected layout on disk:
        skills_dir/
            python_debug/
                SKILL.md          ← required — frontmatter + instructions
                scripts/          ← optional — helpers
                templates/        ← optional — output templates
                resources/        ← optional — reference docs

    Skills come from the repo, so they are untrusted input (B5):
      - a skill is offered to the agent only after the user trusted it (``/skills trust``);
        the trust is pinned to a hash of all its files (``kartrix.skills.trust``), so any change
        needs a new approval;
      - names must be simple identifiers; description/when_to_use are cleaned and capped, since
        they go into the system prompt;
      - symlinked skill folders and SKILL.md files are skipped (they could point outside the repo).
    """

    def __init__(self, skills_dir: Path, workspace_root: Path | None = None) -> None:
        self._skills_dir = skills_dir
        self._root = workspace_root or skills_dir.parent.parent
        self._skills: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Startup: scan disk
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Walk skills_dir and parse every SKILL.md found. Safe to call again to reload."""
        self._skills.clear()
        if not self._skills_dir.is_dir():
            logger.info(f"No skills folder at {self._skills_dir}")
            return

        for skill_dir in sorted(self._skills_dir.iterdir()):
            if skill_dir.is_symlink() or not skill_dir.is_dir():
                continue
            skill_file = skill_dir / SKILL_FILENAME
            if skill_file.is_symlink() or not skill_file.is_file():
                logger.warning(f"Skipping {skill_dir.name}: no SKILL.md found")
                continue
            try:
                skill = _read_skill(skill_dir, skill_file)
            except Exception as e:
                logger.error(f"Failed to load skill from {skill_dir.name}: {e}")
                continue
            if skill is None:
                continue
            self._skills[skill.name] = skill
            logger.info(f"Found skill: {skill.name} ({self.status(skill.name)})")

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    def status(self, name: str) -> Status:
        skill = self._get(name)
        trusted = trust.trusted_digests(self._root).get(name)
        if trusted is None:
            return "untrusted"
        return "trusted" if trusted == skill.digest else "changed"

    def trust(self, name: str) -> Skill:
        skill = self._get(name)
        skill.digest = trust.skill_digest(skill.skill_dir)  # what the user approves is what's on disk now
        trust.trust(self._root, name, skill.digest)
        return skill

    def untrust(self, name: str) -> None:
        self._get(name)
        trust.untrust(self._root, name)

    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    # ------------------------------------------------------------------
    # Called at agent build time → appended to the system prompt
    # ------------------------------------------------------------------

    def build_skills_prompt(self) -> str:
        """Metadata of the *trusted* skills only (name, description, when_to_use); the agent loads
        the full instructions on demand with load_skill (progressive disclosure)."""
        trusted = [s for s in self._skills.values() if self.status(s.name) == "trusted"]
        if not trusted:
            return ""
        lines = ["=== Available Skills (approved by the user) ==="]
        for skill in trusted:
            line = f"- {skill.name}: {skill.description}"
            if skill.when_to_use:
                line += f" | when_to_use: {skill.when_to_use}"
            lines.append(line)
        lines.append(
            "\nWhen the user's request matches a skill, call load_skill(name) to get the full instructions before proceeding."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Called at query time by the load_skill tool
    # ------------------------------------------------------------------

    def load_skill(self, name: str) -> str:
        """The skill's instructions plus its support files (workspace-relative, for read_file).
        Refused unless the skill is trusted and unchanged since — checked again on every load."""
        skill = self._get(name)
        current = trust.skill_digest(skill.skill_dir)
        trusted = trust.trusted_digests(self._root).get(name)
        if trusted is None:
            return f"Skill '{name}' is not approved. Ask the user to review it and run /skills trust {name}."
        if trusted != current:
            return (
                f"Skill '{name}' changed since the user approved it, so it is not loaded. "
                f"Ask the user to review the changes and run /skills trust {name} again."
            )
        result = skill.body
        support_files = _list_support_files(skill.skill_dir, self._root)
        if support_files:
            result += "\n\n--- Support Files Available ---\n"
            result += "\n".join(f"  {p}" for p in support_files)
            result += (
                "\nYou can read any of these files using the read_file tool if the skill instructions reference them."
            )
        return result

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def _get(self, name: str) -> Skill:
        if name not in self._skills:
            available = ", ".join(self._skills.keys()) or "none"
            raise SkillNotFoundError(f"Skill '{name}' not found. Available skills: {available}")
        return self._skills[name]


# ---------------------------------------------------------------------------
# Module-level helpers (private to this file)
# ---------------------------------------------------------------------------


def _clean_meta(value: object) -> str:
    text, _ = strip_hidden(str(value or ""))
    return visible(" ".join(text.split()))[:_META_CHARS]


def _read_skill(skill_dir: Path, skill_file: Path) -> Skill | None:
    meta, body = _parse_skill_file(skill_file)
    name = str(meta.get("name") or skill_dir.name)
    if not _NAME.match(name):
        logger.warning(f"Skipping skill in {skill_dir.name}: invalid name {visible(name)[:80]!r}")
        return None
    description = _clean_meta(meta.get("description") or "No description provided.")
    when_to_use = _clean_meta(meta.get("when_to_use"))
    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        body=body,
        skill_dir=skill_dir,
        digest=trust.skill_digest(skill_dir),
        findings=scan(f"{description}\n{when_to_use}\n{body}"),
    )


def _parse_skill_file(path: Path) -> tuple[dict, str]:
    """
    Read a SKILL.md and split it into (frontmatter_dict, body_str).
    A valid SKILL.md looks like:
        ---
        name: python_debug
        description: Debug Python errors and tracebacks
        when_to_use: error, traceback, exception
        ---
        You are an expert Python debugger...   ← this is the body

    Without frontmatter the whole file is the body and the folder name becomes the skill name.
    """
    text = path.read_text(encoding="utf-8")

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text.strip()

    meta = _parse_yaml(match.group(1))
    return (meta if isinstance(meta, dict) else {}), match.group(2).strip()


def _parse_yaml(text: str) -> dict:
    import yaml  # already a dependency (config.py)

    return yaml.safe_load(text) or {}


def _list_support_files(skill_dir: Path, workspace_root: Path) -> list[str]:
    """Every regular file in the skill folder except SKILL.md, relative to the workspace root
    (the form read_file expects). Links are skipped."""
    files = []
    for f in sorted(skill_dir.rglob("*")):
        if f.is_file() and not f.is_symlink() and f.name != SKILL_FILENAME:
            try:
                files.append(f.relative_to(workspace_root).as_posix())
            except ValueError:
                files.append(f.as_posix())
    return files
