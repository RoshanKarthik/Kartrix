"""Inputs that reach SQL or other parsers must be neutralised before they get there."""

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


def test_mcp_config_survives_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: ${CWD} was substituted unescaped into JSON, so D:\... broke startup.
    monkeypatch.setenv("CWD", 'D:\\proj\\"quoted"\\new')
    servers = load_mcp_configs()
    assert servers["filesystem"]["args"][-1] == 'D:\\proj\\"quoted"\\new'
