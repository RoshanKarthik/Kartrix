"""The model catalog (kartrix.llm.catalog) and its config validation."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from kartrix.config import CatalogModel, ModelsSettings, ModelTier, Settings, settings
from kartrix.llm import catalog

TIERS = {"free": ModelTier(name="Free"), "byok": ModelTier(name="Your key", own_key=True)}


def _models(*entries: dict[str, Any]) -> ModelsSettings:
    return ModelsSettings(tiers=TIERS, catalog=[CatalogModel(**e) for e in entries])


DEFAULT = {"id": "main", "name": "Main", "provider": "nvidia", "model": "vendor/main", "tier": "free"}
BYOK = {"id": "claude", "name": "Claude", "provider": "anthropic", "model": "claude-x", "tier": "byok"}


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.llm, "provider", "nvidia")
    monkeypatch.setattr(settings.llm, "model", "vendor/main")
    monkeypatch.setattr(settings, "models", _models(BYOK, DEFAULT))


def test_catalog_validation() -> None:
    with pytest.raises(ValueError, match="duplicate id"):
        _models(DEFAULT, DEFAULT)
    with pytest.raises(ValueError, match="unknown tier"):
        _models({**DEFAULT, "tier": "gold"})
    with pytest.raises(ValueError):
        _models({**DEFAULT, "id": "Has Spaces"})
    with pytest.raises(ValueError, match="no entry for the default model"):
        Settings(models={"tiers": {"free": {"name": "Free"}}, "catalog": [{**DEFAULT, "model": "other"}]})


def test_shipped_catalog_lists_the_default_model() -> None:
    entry = catalog.default_entry()
    assert (entry.provider, entry.model) == (settings.llm.provider, settings.llm.model)
    assert len({m.id for m in catalog.entries()}) == len(catalog.entries())


@pytest.mark.usefixtures("configured")
def test_default_first_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    listed = catalog.list_models()
    assert [m.id for m in listed] == ["main", "claude"]
    assert listed[0].default and listed[0].available
    assert not listed[1].available and listed[1].unavailable_reason == "needs ANTHROPIC_API_KEY"
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert catalog.list_models()[0].unavailable_reason == "needs NVIDIA_API_KEY"


@pytest.mark.usefixtures("configured")
def test_own_key_models_never_use_a_servers_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-key")
    local = {m.id: m for m in catalog.list_models()}
    hosted = {m.id: m for m in catalog.list_models(on_server=True)}
    assert local["claude"].available  # a local install: the key in the environment is the user's own
    assert not hosted["claude"].available  # a server: the operator's key must not pay for "your key" models


def test_prices_come_from_budgets(monkeypatch: pytest.MonkeyPatch, configured: None) -> None:
    from kartrix.config import ModelPrice

    monkeypatch.setattr(settings.budgets, "prices", {"claude-x": ModelPrice(input=2, output=10)})
    by_id = {m.id: m for m in catalog.list_models()}
    assert (by_id["claude"].price_input, by_id["claude"].price_output) == (2, 10)
    assert by_id["main"].price_input is None


def test_without_a_catalog_the_llm_model_is_the_only_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "models", ModelsSettings())
    (only,) = catalog.list_models()
    assert only.default and only.model == settings.llm.model and only.tier == "default"
    assert set(catalog.tiers()) == {"default"}


@pytest.mark.usefixtures("configured")
def test_models_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.ui.model_list import local_catalog, render_catalog

    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    console = Console(record=True, width=160)
    render_catalog(console, local_catalog(), "this install")
    text = console.export_text()
    assert "main" in text and "(default)" in text and "available" in text
    assert "needs ANTHROPIC_API_KEY" in text
