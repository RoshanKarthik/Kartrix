import json


def test_stats_json(run):
    run("add", "red apple")
    run("add", "green apple pie")
    code, out = run("stats", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["notes"] == 2
    assert data["words"] == 5
    assert data["top_words"][0] == ["apple", 2]


def test_stats_json_empty(run):
    _, out = run("stats", "--json")
    assert json.loads(out) == {"notes": 0, "words": 0, "top_words": []}


def test_stats_text_unchanged(run):
    run("add", "red apple")
    _, out = run("stats")
    assert out.splitlines()[:2] == ["notes: 1", "words: 2"]
