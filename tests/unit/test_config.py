from pathlib import Path

import pytest

from kartrix.config import ConfigError, Settings, load_settings, load_yaml_strict


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "llm:\n  model: a\nllm:\n  model: b\n")
    with pytest.raises(ConfigError, match="Duplicate key 'llm'"):
        load_yaml_strict(p)


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml_strict(_write(tmp_path, "- just\n- a list\n"))


def test_unknown_key_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARTRIX_CONFIG_FILE", str(_write(tmp_path, "retrieval:\n  top_kk: 3\n")))
    with pytest.raises(ConfigError, match="top_kk"):
        load_settings()


def test_invalid_value_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARTRIX_CONFIG_FILE", str(_write(tmp_path, "retrieval:\n  mode: fuzzy\n")))
    with pytest.raises(ConfigError):
        load_settings()


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KARTRIX_CONFIG_FILE", str(_write(tmp_path, "retrieval:\n  top_k: 3\n")))
    monkeypatch.setenv("KARTRIX_RETRIEVAL__TOP_K", "9")
    assert load_settings().retrieval.top_k == 9


def test_cache_namespace_is_validated() -> None:
    with pytest.raises(ValueError):
        Settings(semantic_cache={"namespace": "bad namespace*"})


def test_shipped_config_is_valid() -> None:
    load_yaml_strict(Path("kartrix/config.yaml"))
    assert load_settings().embeddings.dims == 2048
