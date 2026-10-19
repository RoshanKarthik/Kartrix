"""Per-user settings: the user's config.yaml merged over the built-in one, and provider keys in keys.env
(kartrix.keys) — saved privately, loaded after the real environment, checked without calling real APIs here."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from kartrix import keys
from kartrix.config import load_settings, merge_config


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KARTRIX_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KARTRIX_CONFIG_FILE", raising=False)
    return tmp_path / "home"


def test_merge_config_is_section_by_section() -> None:
    base = {"llm": {"model": "a", "timeout": 90, "retry": {"max_retries": 2}}, "index": {"max_file_kb": 512}}
    merged = merge_config(base, {"llm": {"model": "b", "retry": {"max_retries": 5}}, "tracing": {"enabled": False}})
    assert merged == {
        "llm": {"model": "b", "timeout": 90, "retry": {"max_retries": 5}},
        "index": {"max_file_kb": 512},
        "tracing": {"enabled": False},
    }
    assert base["llm"]["model"] == "a"  # the inputs are not changed


def test_user_config_yaml_overrides_the_built_in_one(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / "config").mkdir(parents=True)
    (home / "config" / "config.yaml").write_text("retrieval:\n  top_k: 7\n", encoding="utf-8")
    loaded = load_settings()
    assert loaded.retrieval.top_k == 7
    assert loaded.retrieval.candidates == 40  # the rest of the section stays as shipped
    own = home / "own.yaml"
    own.write_text("retrieval:\n  top_k: 3\n", encoding="utf-8")
    monkeypatch.setenv("KARTRIX_CONFIG_FILE", str(own))
    assert load_settings().retrieval.top_k == 3  # an explicit config file is the whole config


def test_a_bad_user_config_is_reported(home: Path) -> None:
    from kartrix.config import ConfigError

    (home / "config").mkdir(parents=True)
    (home / "config" / "config.yaml").write_text("retrieval:\n  top_kk: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top_kk"):
        load_settings()


def test_keys_file_round_trip(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    keys.save("groq_api_key", "  gsk_abcdefghijklmnop  ")
    assert keys.read() == {"GROQ_API_KEY": "gsk_abcdefghijklmnop"}
    assert os.environ["GROQ_API_KEY"] == "gsk_abcdefghijklmnop"  # usable at once
    keys.save("OPENROUTER_API_KEY", "sk-or-v1-x")
    assert set(keys.read()) == {"GROQ_API_KEY", "OPENROUTER_API_KEY"}
    if sys.platform != "win32":
        assert (keys.keys_file().stat().st_mode & 0o777) == 0o600
    assert keys.remove("GROQ_API_KEY") and "GROQ_API_KEY" not in os.environ
    assert not keys.remove("GROQ_API_KEY")
    assert keys.keys_file().is_relative_to(home)
    for bad_name, bad_value in (("bad name", "x"), ("OK", ""), ("OK", "two\nlines")):
        with pytest.raises(ValueError):
            keys.save(bad_name, bad_value)


def test_mask() -> None:
    assert keys.mask("nvapi-1234567890abcdef") == "nvapi-…cdef"
    assert "1234" not in keys.mask("short-1234")


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(200, True), (401, False), (403, False), (500, None), (httpx.ConnectError("offline"), None)],
)
def test_validate(answer: Any, expected: bool | None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(url: str, **kwargs: Any) -> _Response:
        calls.append(url)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(httpx, "get", fake)
    monkeypatch.setattr(httpx, "post", fake)
    for name in ("NVIDIA_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "HF_TOKEN"):
        ok, _ = keys.validate(name, "secret-value")
        assert ok is expected, name
    assert any("integrate.api.nvidia.com/v1/embeddings" in u for u in calls)  # NVIDIA's model list is public
    assert keys.validate("SOMETHING_ELSE", "x")[0] is None


def test_keys_file_never_overrides_the_environment(home: Path) -> None:
    (home / "config").mkdir(parents=True)
    (home / "config" / "keys.env").write_text("PROBE_FROM_KEYS_FILE=from-file\n", encoding="utf-8")
    code = "import os, kartrix.config; print(os.environ.get('PROBE_FROM_KEYS_FILE'))"
    env = {**os.environ, "KARTRIX_HOME": str(home)}
    env.pop("PROBE_FROM_KEYS_FILE", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "from-file"
    env["PROBE_FROM_KEYS_FILE"] = "from-env"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "from-env"
