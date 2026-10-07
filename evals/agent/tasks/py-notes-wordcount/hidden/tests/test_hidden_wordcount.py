import pytest

from notes.textstats import word_count


@pytest.mark.parametrize(
    ("text", "words"),
    [("two  words", 2), ("a\tb\nc", 3), ("  padded  ", 1), ("", 0), ("   ", 0), ("one", 1)],
)
def test_word_count(text, words):
    assert word_count(text) == words


def test_stats_counts_words(run):
    run("add", "first   note\nhere")
    _, out = run("stats")
    assert "words: 3" in out.splitlines()
