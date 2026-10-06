"""Secret detection/redaction (B6) and the places it is applied: logs and file-tool guards.

Fake credentials are assembled from pieces so this file itself doesn't trip secret scanners.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from kartrix.observability.logger import JSONFormatter, RedactingFormatter
from kartrix.security import secrets
from kartrix.security import workspace as ws_mod
from kartrix.security.secrets import find_secrets, load_known_secrets, redact, redact_with_count, shannon_entropy
from kartrix.security.workspace import set_workspace
from kartrix.tools.filesystem_tools import edit_file, write_file

GITHUB = "ghp_" + "a1B2c3D4e5" * 4
AWS = "AKIA" + "IOSFODNN7EXAMPL0"
OPENAI = "sk-" + "proj-" + "abcDEF1234567890abcdefXYZ"
ANTHROPIC = "sk-" + "ant-" + "api03-abcDEF1234567890abcdef"
NVIDIA = "nv" + "api-" + "Zx81Yw72Vu63Ts54Rq45Po36"
HF = "hf" + "_" + "AbCdEfGhIjKlMnOpQrStUvWxYz123456"
GOOGLE = "AI" + "za" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
STRIPE = "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc"
SLACK = "xo" + "xb-" + "123456789012-abcdefABCDEF"
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9.eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA " + "PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (f"token = {GITHUB}", "github-token"),
        (f"export AWS_ACCESS_KEY_ID={AWS}", "aws-access-key"),
        (f"OPENAI_API_KEY={OPENAI}", "openai-key"),
        (f"key: {ANTHROPIC}", "anthropic-key"),
        (f"client = Client({NVIDIA!r})", "nvidia-key"),
        (f"HF_TOKEN={HF}", "huggingface-token"),
        (f"maps_key = '{GOOGLE}'", "google-api-key"),
        (f"stripe.api_key = '{STRIPE}'", "stripe-key"),
        (f"SLACK={SLACK}", "slack-token"),
        (f"Cookie: session={JWT}", "jwt"),
        (PEM, "private-key"),
        ("Authorization: Bearer abcDEF1234567890xyz", "auth-header"),
        ('aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLE1"', "assignment"),
        ('"db_password": "Tr0ub4dor&3-horse"', "assignment"),
    ],
)
def test_rules_find_secrets(text: str, rule: str) -> None:
    out = redact(text)
    assert f"[REDACTED:{rule}]" in out, out


def test_url_password_only_password_replaced() -> None:
    url = "postgresql+asyncpg://kartrix:" + "S3cr3tP4ss" + "@127.0.0.1:5432/kartrix"
    assert redact(url) == "postgresql+asyncpg://kartrix:[REDACTED:url-password]@127.0.0.1:5432/kartrix"


@pytest.mark.parametrize(
    "text",
    [
        "password = get_password()",
        "password: ${DB_PASSWORD}",
        "api_key = os.environ['API_KEY']",
        "token = settings.github_token",
        "secret_name = 'my_secret_value_name'",
        "def get_token(self): return self._token",
        "max_tokens=3000",
        "token_type = 'Bearer'",
        "sha256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'",
        "commit 6d1e657 feat: add workspace jail",
        'PASSWORD_MIN_LENGTH = 12  # "password" appears, value is short',
        "https://github.com/owner/repo/pull/12",
        "password = 'changeme-in-production'",
        "redis://127.0.0.1:6379/0",
        "file_secret_settings: PydanticBaseSettingsSource,",
        "api_key: Optional.SecretStr = None",
    ],
)
def test_ordinary_code_untouched(text: str) -> None:
    assert redact(text) == text


def test_redaction_is_idempotent_and_counts() -> None:
    text = f"a={GITHUB}\nb={AWS}\n"
    once, n = redact_with_count(text)
    assert n == 2 and redact(once) == once
    assert [f.rule for f in find_secrets(text)] == ["github-token", "aws-access-key"]


def test_known_values_redacted_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = list(secrets._known)
    try:
        load_known_secrets(
            {
                "MY_SERVICE_TOKEN": "plain-lowercase-value",  # no rule would catch this format
                "DATABASE_URL": "postgresql://u:" + "hunter2hunter2" + "@h/db",
                "HOME": "/home/someone",
            }
        )
        assert redact("echo plain-lowercase-value") == "echo [REDACTED:env]"
        assert "hunter2hunter2" not in redact("password is hunter2hunter2")
        assert redact("/home/someone") == "/home/someone"
    finally:
        secrets._known[:] = previous


def test_entropy() -> None:
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("abcdefgh") == 3


# ── logs ──────────────────────────────────────────────────────────────


def _record(msg: str, **extra: object) -> logging.LogRecord:
    record = logging.makeLogRecord({"msg": msg, "levelname": "INFO", "name": "kartrix.test"})
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_log_lines_are_redacted() -> None:
    line = JSONFormatter().format(_record(f"calling with {OPENAI}", argv=["curl", "-H", f"x-api-key: {GITHUB}"]))
    payload = json.loads(line)
    assert OPENAI not in line and GITHUB not in line
    assert payload["msg"] == "calling with [REDACTED:openai-key]"
    text = RedactingFormatter("%(message)s").format(_record(f"key {AWS}"))
    assert text == "key [REDACTED:aws-access-key]"


# ── file tools never write placeholders back ──────────────────────────


@pytest.fixture
def root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "ws"
    root.mkdir()
    previous = ws_mod._current
    set_workspace(root)
    yield root
    ws_mod._current = previous


def test_write_file_refuses_to_clobber_redacted_secrets(root: Path) -> None:
    (root / "settings.py").write_text(f"DEBUG = True\nGITHUB_TOKEN = '{GITHUB}'\n")
    out = write_file.invoke(
        {"file_path": "settings.py", "content": "DEBUG = False\nGITHUB_TOKEN = '[REDACTED:github-token]'\n"}
    )
    assert "contains secrets" in out
    assert GITHUB in (root / "settings.py").read_text()
    # a brand-new file may mention a placeholder (e.g. docs or tests about redaction)
    assert write_file.invoke({"file_path": "notes.md", "content": "shows [REDACTED:jwt]"}).startswith("Created")


def test_edit_file_around_redacted_secrets(root: Path) -> None:
    (root / "settings.py").write_text(f"DEBUG = True\nGITHUB_TOKEN = '{GITHUB}'\n")
    seen = "GITHUB_TOKEN = '[REDACTED:github-token]'"
    assert "redacted secret" in edit_file.invoke({"file_path": "settings.py", "old_string": seen, "new_string": "X"})
    assert "can't be written back" in edit_file.invoke(
        {"file_path": "settings.py", "old_string": "DEBUG = True", "new_string": seen}
    )
    assert "replaced 1" in edit_file.invoke(
        {"file_path": "settings.py", "old_string": "DEBUG = True", "new_string": "DEBUG = False"}
    )
    assert (root / "settings.py").read_text() == f"DEBUG = False\nGITHUB_TOKEN = '{GITHUB}'\n"
