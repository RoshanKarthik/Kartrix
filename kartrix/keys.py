"""The user's provider keys (``kartrix keys``, the first-run prompt): saved in ``<user config dir>/keys.env``
(owner-only), loaded by :mod:`kartrix.config` after the real environment and a development checkout's ``.env``.

Each key can be checked with one cheap call that changes nothing (``validate``) — listing models, or for NVIDIA a
one-word embedding, because its model list is public.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import kartrix.config  # noqa: F401 — loads .env and keys.env, so os.environ shows every configured key
from kartrix.paths import user_config_dir


@dataclass(frozen=True)
class KeyInfo:
    name: str
    provider: str
    required: bool
    where: str  # where to get one, free


KNOWN = (
    KeyInfo("NVIDIA_API_KEY", "NVIDIA", True, "https://build.nvidia.com (free; used for chat and the code index)"),
    KeyInfo("GROQ_API_KEY", "Groq", False, "https://console.groq.com/keys (free plan, very fast)"),
    KeyInfo("OPENROUTER_API_KEY", "OpenRouter", False, "https://openrouter.ai/keys (free :free models)"),
    KeyInfo("GEMINI_API_KEY", "Gemini", False, "https://aistudio.google.com/apikey (free tier may train on prompts)"),
    KeyInfo("HF_TOKEN", "Hugging Face", False, "https://huggingface.co/settings/tokens (fallback model)"),
)
_BY_NAME = {k.name: k for k in KNOWN}


def keys_file() -> Path:
    return user_config_dir() / "keys.env"


def known(name: str) -> KeyInfo | None:
    return _BY_NAME.get(name.upper())


def read() -> dict[str, str]:
    """The keys saved in the user's keys file (not the environment)."""
    from dotenv import dotenv_values

    path = keys_file()
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


def _write(values: dict[str, str]) -> None:
    path = keys_file()
    lines = ["# Kartrix provider keys (kartrix keys). Keep this file private.\n"]
    lines += [f"{name}={value}\n" for name, value in sorted(values.items())]
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp, path)


def save(name: str, value: str) -> None:
    """Save (or replace) one key and use it in this process right away."""
    name, value = name.strip().upper(), value.strip()
    if not name.replace("_", "").isalnum() or not value or any(c in value for c in "\r\n"):
        raise ValueError("a key needs a NAME made of letters, digits and _ and a one-line value")
    values = read()
    values[name] = value
    _write(values)
    os.environ[name] = value


def remove(name: str) -> bool:
    values = read()
    if values.pop(name.upper(), None) is None:
        return False
    _write(values)
    os.environ.pop(name.upper(), None)
    return True


def mask(value: str) -> str:
    return value[:6] + "…" + value[-4:] if len(value) > 14 else "…" + value[-2:]


def validate(name: str, value: str, timeout: float = 20.0) -> tuple[bool | None, str]:
    """(True, "") valid · (False, why) rejected by the provider · (None, why) couldn't check (offline, unknown key)."""
    import httpx

    name = name.upper()
    try:
        if name == "NVIDIA_API_KEY":
            from kartrix.config import settings

            r = httpx.post(
                "https://integrate.api.nvidia.com/v1/embeddings",
                headers={"Authorization": f"Bearer {value}"},
                json={"model": settings.embeddings.model, "input": ["ok"], "input_type": "query"},
                timeout=timeout,
            )
        elif name == "GROQ_API_KEY":
            r = httpx.get(
                "https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {value}"}, timeout=timeout
            )
        elif name == "OPENROUTER_API_KEY":
            r = httpx.get(
                "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {value}"}, timeout=timeout
            )
        elif name == "GEMINI_API_KEY":
            r = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": value},
                timeout=timeout,
            )
        elif name == "HF_TOKEN":
            r = httpx.get(
                "https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {value}"}, timeout=timeout
            )
        else:
            return None, "no check for this key"
    except httpx.HTTPError as e:
        return None, f"couldn't reach the provider ({type(e).__name__})"
    if r.status_code in (401, 403):
        return False, "the provider rejected this key"
    if r.status_code >= 400:
        return None, f"the provider answered HTTP {r.status_code}"
    return True, ""


def missing_required() -> list[KeyInfo]:
    """Keys the configured models need (main/router/judge and embeddings providers) but that aren't set."""
    from kartrix.config import settings
    from kartrix.llm.catalog import PROVIDER_KEYS

    needed = {
        k for k in (PROVIDER_KEYS.get(settings.llm.provider), PROVIDER_KEYS.get(settings.embeddings.provider)) if k
    }
    missing = [n for n in sorted(needed) if n and not os.environ.get(n, "").strip()]
    return [_BY_NAME.get(n) or KeyInfo(n, n, True, "your provider's dashboard") for n in missing]


def first_run(interactive: bool | None = None) -> bool:
    """Ask for the keys Kartrix can't start without (once; they are saved for every repo). False if still missing."""
    import getpass
    import sys

    missing = missing_required()
    if not missing:
        return True
    interactive = sys.stdin.isatty() if interactive is None else interactive
    if not interactive:
        for info in missing:
            print(f"kartrix: {info.name} is not set — run `kartrix keys set {info.name}` (free: {info.where})")
        return False
    print("Welcome to Kartrix. It needs one free key to start (saved once, used in every repo):")
    for info in missing:
        print(f"\n{info.provider}: get a free key at {info.where}")
        for _ in range(3):
            value = getpass.getpass(f"Paste your {info.name} (hidden as you type; Enter to cancel): ").strip()
            if not value:
                return False
            ok, why = validate(info.name, value)
            if ok is False:
                print(f"  {why} — try again.")
                continue
            save(info.name, value)
            print("  Saved." if ok else f"  Saved, but not checked ({why}).")
            break
        else:
            return False
    print("\nOptional, free extras later: `kartrix keys set GROQ_API_KEY` (fast), OPENROUTER_API_KEY, GEMINI_API_KEY.")
    return True
