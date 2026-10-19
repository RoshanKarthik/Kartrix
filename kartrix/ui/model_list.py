"""``kartrix models``: the model catalog as a table — from the server after ``kartrix login``, else local."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table


def local_catalog() -> dict[str, Any]:
    """The local catalog in the API's shape (GET /v1/models)."""
    from kartrix.llm import catalog

    return {
        "default": catalog.default_entry().id,
        "tiers": [{"id": key, **tier.model_dump()} for key, tier in catalog.tiers().items()],
        "models": [asdict(m) for m in catalog.list_models()],
    }


def _price(model: dict[str, Any]) -> str:
    if model.get("price_input") is None:
        return "?"
    if model["price_input"] == 0 and model["price_output"] == 0:
        return "free"
    return f"${model['price_input']:g} / ${model['price_output']:g}"


def render_catalog(console: Console, data: dict[str, Any], source: str, current: str | None = None) -> None:
    tiers = {t["id"]: t for t in data.get("tiers", [])}
    table = Table(title=f"Models ({escape(source)})", title_justify="left", header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Model")
    table.add_column("Tier")
    table.add_column("$ in/out per 1M", justify="right", no_wrap=True)
    table.add_column("Status")
    for model in data.get("models", []):
        tier = tiers.get(model["tier"], {"name": model["tier"]})
        reason = model["unavailable_reason"] or "unavailable"
        status = "[green]available[/green]" if model["available"] else f"[dim]{escape(reason)}[/dim]"
        name = escape(model["name"]) + (" [bold](default)[/bold]" if model["default"] else "")
        if model["id"] == current:
            name = "[bold green]▶[/bold green] " + name
        table.add_row(escape(model["id"]), name, escape(tier["name"]), _price(model), status)
    console.print(table)
