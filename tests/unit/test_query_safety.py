"""Inputs that reach SQL or other parsers must be neutralised before they get there."""

import pytest

from kartrix.context.indexers.pg_index import split_identifiers
from kartrix.context.retrievers.pg_hybrid import build_tsquery


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
