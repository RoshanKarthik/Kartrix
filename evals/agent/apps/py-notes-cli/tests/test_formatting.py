from notes.formatting import fmt_note, normalise_tags


def test_normalise_tags():
    assert normalise_tags(["Work", "home", "work", " "]) == ["home", "work"]


def test_fmt_note_without_tags():
    assert fmt_note({"id": 7, "text": "hello", "tags": []}) == "#7 hello"
