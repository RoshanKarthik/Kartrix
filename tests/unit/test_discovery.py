import shutil
import subprocess
from pathlib import Path

import pytest

from kartrix.context.discovery import RepoFilter, discover_files


def _w(root: Path, rel: str, text: str = "x = 1\n", data: bytes | None = None) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        p.write_bytes(data)
    else:
        p.write_text(text, encoding="utf-8")


@pytest.fixture
def tree(repo: Path) -> Path:
    _w(repo, ".gitignore", "*.log\nbuild/\nsecret_*.py\n!secret_ok.py\n/rootonly.py\n")
    for rel in [
        "src/app.py",
        "src/app.log",
        "build/gen.py",
        "secret_a.py",
        "secret_ok.py",
        "rootonly.py",
        "src/rootonly.py",
        "pkg/notes.md",
        "pkg/README.md",
        "pkg/mod.py",
        "pkg/sub/deep.md",
        "build/keep.py",
        "local_only.py",
        "node_modules/x/index.js",
        "certs/server.pem",
        "config.yaml",
    ]:
        _w(repo, rel)
    _w(repo, "pkg/.gitignore", "*.md\n!README.md\n")  # nested rules + re-include
    _w(repo, "build/.gitignore", "!keep.py\n")  # can't re-include inside ignored dir
    _w(repo, ".git/info/exclude", "local_only.py\n")
    _w(repo, ".env", "TOKEN=x\n")
    _w(repo, "big.py", "x" * (600 * 1024))  # over max_file_kb
    _w(repo, "bin.py", data=b"abc\x00def")  # binary
    _w(repo, "image.png", data=b"\x89PNG")  # unknown extension
    _w(repo, "empty.py", "")
    return repo


EXPECTED = ["config.yaml", "empty.py", "pkg/README.md", "pkg/mod.py", "secret_ok.py", "src/app.py", "src/rootonly.py"]


def test_discovery_honours_gitignore_and_excludes(tree: Path) -> None:
    got = sorted(f.relative_to(tree.resolve()).as_posix() for f in discover_files(tree))
    assert got == EXPECTED


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_discovery_matches_git(tree: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tree, check=True)
    _w(tree, ".git/info/exclude", "local_only.py\n")  # git init may have rewritten it
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=tree, capture_output=True, text=True, check=True
    ).stdout.split()
    # git doesn't know about our size/binary/extension/secret filters; apply them for the comparison
    git_view = {f for f in out if Path(f).suffix in {".py", ".md", ".yaml"} and f not in {"big.py", "bin.py"}}
    assert git_view == set(EXPECTED)


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("src/app.py", True),
        ("pkg/README.md", True),
        ("pkg/notes.md", False),
        ("pkg/sub/deep.md", False),
        ("build/keep.py", False),
        ("secret_a.py", False),
        ("big.py", False),
        ("bin.py", False),
        (".env", False),
        ("../outside.py", False),
    ],
)
def test_is_indexable_single_file(tree: Path, rel: str, expected: bool) -> None:
    assert RepoFilter.load(tree).is_indexable(tree / rel) is expected


def test_passes_rules_works_for_deleted_paths(tree: Path) -> None:
    flt = RepoFilter.load(tree)
    assert flt.passes_rules(tree / "src/never_existed.py")
    assert not flt.passes_rules(tree / ".git/hooks/x.py")
    assert not flt.passes_rules(tree / "src/never_existed.log")
