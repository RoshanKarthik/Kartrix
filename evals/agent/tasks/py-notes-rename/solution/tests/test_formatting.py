from notes.formatting import format_note, normalise_tags


def test_normalise_tags():
    assert normalise_tags(["Work", "home", "work", " "]) == ["home", "work"]


def test_format_note_without_tags():
    assert format_note({"id": 7, "text": "hello", "tags": []}) == "#7 hello"
