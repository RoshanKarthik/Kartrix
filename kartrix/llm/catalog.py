"""The model catalog: the models users can pick, grouped in tiers (config ``models``).

Every entry says whether it can run here: a server-paid model needs its provider's key in the
server's environment; an ``own_key`` tier model needs the user's own key — on a local install the
provider key in the environment is the user's own; on a server it never is (the operator's key must not
pay for "your key" models), so those wait for per-account keys (roadmap 2.3). The default
is the entry matching ``llm.provider`` / ``llm.model``; with no catalog configured that model is the
only entry.

Imports nothing heavy: listing models must not load any provider SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from kartrix.config import CatalogModel, ModelTier, settings

# Env var holding each provider's API key (the same names the model factory requires).
PROVIDER_KEYS = {
    "nvidia": "NVIDIA_API_KEY",
    "huggingface": "HF_TOKEN",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_DEFAULT_TIER = ModelTier(name="Default")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    provider: str
    model: str
    tier: str
    description: str
    context_tokens: int | None
    default: bool
    available: bool
    unavailable_reason: str | None
    price_input: float | None  # USD per million tokens (budgets.prices); None = unknown
    price_output: float | None


def entries() -> list[CatalogModel]:
    """The configured catalog, or the one ``llm`` model when none is configured."""
    if settings.models.catalog:
        return list(settings.models.catalog)
    llm = settings.llm
    return [CatalogModel(id="default", name=llm.model, provider=llm.provider, model=llm.model, tier="default")]


def tiers() -> dict[str, ModelTier]:
    return dict(settings.models.tiers) if settings.models.catalog else {"default": _DEFAULT_TIER}


def find(model_id: str) -> CatalogModel | None:
    return next((m for m in entries() if m.id == model_id), None)


def default_entry() -> CatalogModel:
    llm = settings.llm
    found = next((m for m in entries() if m.provider == llm.provider and m.model == llm.model), None)
    assert found is not None, "config validation guarantees the default is in the catalog"  # noqa: S101
    return found


def _availability(entry: CatalogModel, tier: ModelTier, on_server: bool) -> tuple[bool, str | None]:
    key = PROVIDER_KEYS.get(entry.provider)
    if tier.own_key and on_server:
        return False, "own key: not on servers yet"
    if key is None or os.environ.get(key, "").strip():
        return True, None
    return False, f"needs {key}"


def list_models(*, on_server: bool = False) -> list[ModelInfo]:
    """Every catalog model with its availability and price, default first, then in catalog order.
    ``on_server``: listed for a server's users (their own-key models need per-account keys)."""
    tier_map = tiers()
    default_id = default_entry().id
    prices = settings.budgets.prices
    out = []
    for entry in entries():
        tier = tier_map.get(entry.tier, _DEFAULT_TIER)
        available, reason = _availability(entry, tier, on_server)
        price = prices.get(entry.model)
        out.append(
            ModelInfo(
                id=entry.id,
                name=entry.name,
                provider=entry.provider,
                model=entry.model,
                tier=entry.tier,
                description=entry.description,
                context_tokens=entry.context_tokens,
                default=entry.id == default_id,
                available=available,
                unavailable_reason=reason,
                price_input=price.input if price else None,
                price_output=price.output if price else None,
            )
        )
    out.sort(key=lambda m: not m.default)  # stable: the rest keep catalog order
    return out
