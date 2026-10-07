from notes.formatting import fmt_note, normalise_tags


def test_normalise_tags():
    assert normalise_tags(["Work", "home", "work", " "]) == ["home", "work"]


def test_fmt_note_without_tags():
    assert fmt_note({"id": 7, "text": "hello", "tags": []}) == "#7 hello"


def test_fmt_note_with_tags():
    note = {"id": 3, "text": "Quarterly report", "tags": ["Work", "home"]}
    assert fmt_note(note) == "#3 [home, work] Quarterly report"
