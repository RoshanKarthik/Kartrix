"""Inputs that reach SQL or other parsers must be neutralised before they get there."""

from pathlib import Path

import pytest

from kartrix.context.indexers.pg_index import split_identifiers
from kartrix.context.retrievers.pg_hybrid import build_tsquery
from kartrix.mcp.mcp_config import load_mcp_configs


@pytest.mark.parametrize(
    "query",
    ["a & !b | (c:*)", "x' OR 1=1 --", "<-> foo <2> bar", "'); DROP TABLE code_chunks; --", "\\ \x00 :* !"],
)
def test_tsquery_contains_only_words_and_or(query: str) -> None:
    tsq = build_tsquery(query)
    for term in tsq.split(" | ") if tsq else []:
        assert term.isalnum(), tsq


def test_tsquery_splits_camel_case_and_dedupes() -> None:
    assert build_tsquery("parseConfigFile parseConfigFile") == "parseconfigfile | config | file | parse"


def test_tsquery_empty_for_symbols_only() -> None:
    assert build_tsquery("?? !! &&") == ""


def test_split_identifiers() -> None:
    assert split_identifiers("HTTPServerError getX snake_case") == "error get http server x"


def test_mcp_config_survives_windows_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Regression: ${CWD} was substituted unescaped into JSON, so D:\... broke startup.
    config = tmp_path / "mcp_servers.json"
    config.write_text('{"mcp_servers": {"srv": {"command": "x", "args": ["${CWD}"]}}}')
    monkeypatch.setattr("kartrix.mcp.mcp_config._CONFIG_PATH", config)
    monkeypatch.setenv("CWD", 'D:\\proj\\"quoted"\\new')
    servers = load_mcp_configs()
    assert servers["srv"]["args"][-1] == 'D:\\proj\\"quoted"\\new'
