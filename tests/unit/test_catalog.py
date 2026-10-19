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


# ── picking the main model (roadmap 2.2) ──────────────────────────────


@pytest.fixture
def reset_selection() -> Any:
    from kartrix.llm.factory import select_main_model

    yield
    select_main_model(None)


@pytest.mark.usefixtures("reset_selection")
def test_selected_main_model_is_used_for_the_main_role_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.config import LLMModelRef
    from kartrix.llm import factory

    built: list[tuple[str, str]] = []
    monkeypatch.setattr(factory, "_build_chat_model", lambda provider, model, **kw: built.append((provider, model)))
    factory.select_main_model(LLMModelRef(provider="anthropic", model="claude-x"))
    factory.get_chat_model("main")
    factory.get_chat_model("router")
    factory.get_chat_model("judge")
    assert built[0] == ("anthropic", "claude-x")
    assert built[1] == (settings.llm.provider, settings.llm.effective_router_model)
    assert built[2] == (settings.llm.provider, settings.llm.effective_judge_model)
    factory.select_main_model(None)
    assert factory.main_model().model == settings.llm.model


@pytest.mark.usefixtures("reset_selection")
def test_claude_gets_no_sampling_parameters_and_native_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

    from kartrix.config import LLMModelRef
    from kartrix.llm import factory

    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-real-key")
    llm = factory._build_chat_model("anthropic", "claude-opus-5-5", temperature=0, top_p=0.5)
    assert getattr(llm, "temperature", None) is None and getattr(llm, "top_p", None) is None
    assert isinstance(factory.response_format_for(LLMModelRef), ToolStrategy)  # the default (NIM) model
    factory.select_main_model(LLMModelRef(provider="anthropic", model="claude-haiku-5-5"))
    assert isinstance(factory.response_format_for(LLMModelRef), ProviderStrategy)
    assert isinstance(factory.response_format_for(LLMModelRef, "router"), ToolStrategy)  # router: llm.provider


@pytest.mark.usefixtures("configured", "reset_selection")
def test_select_model_checks_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.core.session import _select_model
    from kartrix.llm.factory import main_model

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="unknown model 'nope'"):
        _select_model("nope")
    with pytest.raises(ValueError, match="needs ANTHROPIC_API_KEY"):
        _select_model("claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-real-key")
    assert _select_model("claude") == "claude"
    assert main_model().model == "claude-x"
    assert _select_model(None) == "main" and main_model().model == "vendor/main"


def test_headless_spec_takes_a_model(tmp_path: Any) -> None:
    from kartrix.headless.spec import load_spec

    path = tmp_path / "spec.yaml"
    path.write_text("task: fix it\nmodel: gemma-4-31b\n", encoding="utf-8")
    assert load_spec(path).model == "gemma-4-31b"


# ── free-tier providers (OpenAI-compatible endpoints) ─────────────────


@pytest.mark.parametrize(
    ("provider", "host"),
    [
        ("groq", "api.groq.com"),
        ("openrouter", "openrouter.ai"),
        ("gemini", "generativelanguage.googleapis.com"),
    ],
)
def test_openai_compatible_providers(provider: str, host: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from kartrix.llm import factory

    llm = factory._build_chat_model(provider, "some/model", temperature=0)
    assert host in str(llm.openai_api_base)  # type: ignore[attr-defined]
    assert llm.model_name == "some/model"  # type: ignore[attr-defined]
    monkeypatch.delenv(catalog.PROVIDER_KEYS[provider])
    with pytest.raises(factory.ProviderConfigError, match=catalog.PROVIDER_KEYS[provider]):
        factory._build_chat_model(provider, "some/model")


def test_shipped_free_models_are_flagged_and_free() -> None:
    by_id = {m.id: m for m in catalog.list_models()}
    for model_id in ("groq-gpt-oss-120b", "gemini-2.5-flash", "nemotron-ultra-free", "laguna-s-free"):
        assert by_id[model_id].price_input == 0 and by_id[model_id].price_output == 0
    assert by_id["gemini-2.5-flash"].trains_on_inputs
    assert not by_id["groq-gpt-oss-120b"].trains_on_inputs
    assert all(m.model.endswith(":free") for m in by_id.values() if m.provider == "openrouter")  # never a paid model
