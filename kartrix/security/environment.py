"""Environment for commands the agent runs: Kartrix's own secrets are stripped (B2).

``.env`` is loaded into Kartrix's process (LLM keys, DATABASE_URL, REDIS_URL, GITHUB_TOKEN).
Child processes must not inherit them, or ``printenv`` / a test script / a postinstall
hook could print them back to the model or send them elsewhere.
"""

from __future__ import annotations

import os
import re

from kartrix.config import settings

_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASSPHRASE|CREDENTIAL|AUTH|COOKIE|PRIVATE", re.I)
_SECRET_NAMES = {"DATABASE_URL", "TEST_DATABASE_URL", "REDIS_URL"}
_URL_WITH_PASSWORD = re.compile(r"://[^/\s:@]*:[^/\s@]+@")


def is_secret_var(name: str, value: str) -> bool:
    upper = name.upper()
    return (
        upper in _SECRET_NAMES
        or upper.startswith("KARTRIX_")
        or bool(_SECRET_NAME.search(name))
        or bool(_URL_WITH_PASSWORD.search(value))
    )


def scrubbed_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Copy of the environment without secrets (``permissions.env_passthrough`` names are kept)."""
    env = dict(os.environ if base is None else base)
    keep = {n.upper() for n in settings.permissions.env_passthrough}
    return {k: v for k, v in env.items() if k.upper() in keep or not is_secret_var(k, v)}
