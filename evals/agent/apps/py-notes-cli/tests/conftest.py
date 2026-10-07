import io

import pytest

from notes.cli import main
from notes.store import NoteStore


@pytest.fixture
def store(tmp_path):
    return NoteStore(tmp_path / "notes.json")


@pytest.fixture
def run(store):
    def _run(*argv):
        out = io.StringIO()
        code = main(list(argv), store=store, out=out)
        return code, out.getvalue()

    return _run
