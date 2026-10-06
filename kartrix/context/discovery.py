""".gitignore-aware discovery of indexable files (A9).

Honours, like git does: every ``.gitignore`` in the tree (deeper files override
shallower ones, ``!`` re-includes), ``.git/info/exclude``, and the rule that files
inside an ignored directory can't be re-included. On top of that, ``index.exclude``
from config is always applied (build dirs, secrets) even in folders that aren't
git repos or have no .gitignore.

Only regular files with a known extension, under the size cap and not binary are
returned. Symlinks are never followed, so indexing can't escape the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pathspec import GitIgnoreSpec

from kartrix.config import settings
from kartrix.context.indexers.code_parser import ALL_EXTENSIONS
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_BINARY_SNIFF_BYTES = 8192


def _read_spec(path: Path) -> GitIgnoreSpec | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    spec = GitIgnoreSpec.from_lines(lines)
    return spec if spec.patterns else None


@dataclass
class RepoFilter:
    """Decides whether a path under ``root`` is ignored / indexable.

    ``specs`` maps a directory (relative POSIX path, "" for the root) to the
    compiled ``.gitignore`` found there. Build with :meth:`load`.
    """

    root: Path
    exclude: GitIgnoreSpec
    specs: dict[str, GitIgnoreSpec] = field(default_factory=dict)
    max_bytes: int = 512 * 1024
    _loaded: set[str] = field(default_factory=lambda: {""})

    @classmethod
    def load(cls, root: str | Path) -> RepoFilter:
        root = Path(root).resolve()
        cfg = settings.index
        flt = cls(root=root, exclude=GitIgnoreSpec.from_lines(cfg.exclude), max_bytes=cfg.max_file_kb * 1024)
        # .git/info/exclude has the lowest priority of the repo-level rules, so it is
        # merged in front of the root .gitignore (later patterns win).
        info_exclude = root / ".git" / "info" / "exclude"
        root_lines: list[str] = []
        for src in (info_exclude, root / ".gitignore"):
            if src.is_file():
                root_lines += src.read_text(encoding="utf-8", errors="ignore").splitlines()
        if root_lines:
            flt.specs[""] = GitIgnoreSpec.from_lines(root_lines)
        return flt

    def rel(self, path: str | Path) -> str | None:
        """POSIX path relative to root, or None if ``path`` is outside the repo."""
        # abspath, not resolve(): a symlink is judged by where it sits, not where it points.
        p = Path(os.path.abspath(self.root / path))
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def add_gitignore(self, rel_dir: str) -> None:
        if rel_dir in self._loaded:
            return  # the root one is loaded in load()
        self._loaded.add(rel_dir)
        spec = _read_spec(self.root / rel_dir / ".gitignore")
        if spec is not None:
            self.specs[rel_dir] = spec

    def is_ignored(self, rel_path: str, is_dir: bool) -> bool:
        """Apply config excludes, then the closest .gitignore that has an opinion."""
        target = rel_path + "/" if is_dir else rel_path
        if self.exclude.match_file(target):
            return True
        parts = rel_path.split("/")
        # Deepest .gitignore first; the first one with a matching pattern decides.
        for depth in range(len(parts) - 1, -1, -1):
            base = "/".join(parts[:depth])
            spec = self.specs.get(base)
            if spec is None:
                continue
            sub = target[len(base) + 1:] if base else target
            result = spec.check_file(sub)
            if result.include is not None:
                return result.include
        return False

    def passes_rules(self, path: str | Path) -> bool:
        """True if ``path`` is inside the repo, has an indexable extension and no ignore
        rule (config, .gitignore, or an ignored ancestor dir) excludes it. Doesn't touch
        the file itself, so it also works for deleted paths."""
        rel = self.rel(path)
        if rel is None or rel == "." or Path(rel).suffix.lower() not in ALL_EXTENSIONS:
            return False
        parts = rel.split("/")
        for i in range(1, len(parts)):
            self.add_gitignore("/".join(parts[:i - 1]))
            if self.is_ignored("/".join(parts[:i]), is_dir=True):
                return False
        self.add_gitignore("/".join(parts[:-1]))
        return not self.is_ignored(rel, is_dir=False)

    def is_indexable(self, path: str | Path) -> bool:
        """Full check for a single file (used by the watcher): rules + size/binary/symlink."""
        return self.passes_rules(path) and self._file_ok(self.root / self.rel(path))

    def _file_ok(self, path: Path) -> bool:
        if path.suffix.lower() not in ALL_EXTENSIONS or path.is_symlink():
            return False
        try:
            if not path.is_file() or path.stat().st_size > self.max_bytes:
                return False
            with path.open("rb") as fh:
                return b"\x00" not in fh.read(_BINARY_SNIFF_BYTES)
        except OSError:
            return False


def discover_files(root: str | Path, repo_filter: RepoFilter | None = None) -> list[Path]:
    """Walk ``root`` and return every indexable file (absolute paths, sorted)."""
    flt = repo_filter or RepoFilter.load(root)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(flt.root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(flt.root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        if ".gitignore" in filenames:
            flt.add_gitignore(rel_dir)

        def _rel(name: str) -> str:
            return f"{rel_dir}/{name}" if rel_dir else name

        # Prune in place: ignored dirs (and their contents) are never visited.
        dirnames[:] = sorted(
            d for d in dirnames
            if not os.path.islink(os.path.join(dirpath, d)) and not flt.is_ignored(_rel(d), is_dir=True)
        )
        for name in filenames:
            rel = _rel(name)
            if not flt.is_ignored(rel, is_dir=False) and flt._file_ok(flt.root / rel):
                found.append(flt.root / rel)
    found.sort()
    logger.info("Discovered indexable files", extra={"root": str(flt.root), "files": len(found)})
    return found
