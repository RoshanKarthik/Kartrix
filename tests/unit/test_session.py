import uuid

import pytest

from kartrix.memory.session import InvalidSessionIdError, validate_session_id


def test_accepts_canonical_uuid4() -> None:
    sid = str(uuid.uuid4())
    assert validate_session_id(sid) == sid


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../../etc/passwd",
        "not-a-uuid",
        "{" + str(uuid.uuid4()) + "}",
        "urn:uuid:" + str(uuid.uuid4()),
        str(uuid.uuid4()).replace("-", ""),
        str(uuid.uuid1()),  # wrong version
        "00000000-0000-0000-0000-000000000000",
        str(uuid.uuid4()) + "; DROP TABLE sessions",
    ],
)
def test_rejects_anything_else(bad: str) -> None:
    with pytest.raises(InvalidSessionIdError):
        validate_session_id(bad)


def test_uppercase_is_normalised_to_canonical_form() -> None:
    sid = str(uuid.uuid4())
    assert validate_session_id(sid.upper()) == sid
