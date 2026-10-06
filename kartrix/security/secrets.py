"""Secret detection and redaction (B6).

Used on every tool output before the model sees it, on every log line, on code chunks
before they are embedded/stored, and on semantic-cache entries. Two layers:

1. **Known values** — the secrets Kartrix itself holds (API keys, DB/Redis passwords from
   ``.env``) are replaced wherever they appear, exactly, regardless of format.
2. **Rules** — gitleaks-style patterns for common credential formats, plus a generic
   ``<secret-ish name> = <high-entropy value>`` rule.

A match becomes ``[REDACTED:<rule>]``. Redaction is idempotent (placeholders never match).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from kartrix.security.environment import is_secret_var

# Bump whenever RULES change: indexed chunks are then re-redacted (part of the index version).
RULES_VERSION = 2

PLACEHOLDER = re.compile(r"\[REDACTED:[a-z0-9-]+\]")


def placeholder(rule: str) -> str:
    return f"[REDACTED:{rule}]"


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern[str]
    group: int = 0  # which group holds the secret (0 = whole match)
    check: Callable[[str], bool] | None = None  # extra test on the secret, e.g. entropy


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values())


# Values that look like assignments to secrets but aren't secrets themselves.
_NOT_A_VALUE = re.compile(
    r"^(?:\$|\{|<|%|\(|os\.|process\.|env\.|settings\.|self\.|config\.|getenv|environ|true$|false$|none$|null$)"
    r"|^(?:x{6,}|\*{3,}|changeme|password|secret|example|your[_-]|my[_-]|placeholder|dummy|test|redacted)",
    re.IGNORECASE,
)


def _looks_random(value: str) -> bool:
    if _NOT_A_VALUE.search(value) or "(" in value or "[" in value:
        return False
    if re.fullmatch(r"[A-Za-z_.]+", value):  # identifiers / type names: `x_secret: SettingsSource`
        return False
    # Real keys mix character classes; identifiers like snake_case_names rarely pass both tests.
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"))
    return shannon_entropy(value) >= 3.5 and classes >= 2


_SENSITIVE_NAME = (
    r"[A-Za-z0-9_.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|private[_-]?key"
    r"|client[_-]?secret|auth[_-]?key|credential|signing[_-]?key|encryption[_-]?key)[A-Za-z0-9_.-]*"
)

RULES: tuple[Rule, ...] = (
    Rule(
        "private-key",
        re.compile(
            r"-----BEGIN[A-Z ]*PRIVATE KEY(?: BLOCK)?-----(?:[\s\S]*?-----END[A-Z ]*PRIVATE KEY(?: BLOCK)?-----|[\s\S]*)"
        ),
    ),
    Rule("aws-access-key", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    Rule("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})\b")),
    Rule("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}")),
    Rule("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    Rule("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+")),
    Rule("stripe-key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    Rule("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    Rule("openai-key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")),
    Rule("nvidia-key", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}")),
    Rule("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    Rule("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    Rule("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    Rule("pypi-token", re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}")),
    Rule("sendgrid-key", re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b")),
    Rule("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # scheme://user:PASSWORD@host — only the password is replaced.
    Rule("url-password", re.compile(r"(?<=://)[^/\s:@]*:([^/\s@]+)(?=@)"), group=1),
    Rule(
        "auth-header",
        re.compile(
            r"(?i)\b(?:authorization|x-api-key|api-key)\b[\"']?\s*[:=]\s*[\"']?(?:bearer |basic |token )?"
            r"([A-Za-z0-9._~+/=-]{12,})"
        ),
        group=1,
    ),
    Rule(
        "assignment",
        re.compile(rf"(?i)\b{_SENSITIVE_NAME}[\"']?\s*(?::=|=>|[:=])\s*[\"'`]?([^\s\"'`,;]{{8,}})"),
        group=1,
        check=_looks_random,
    ),
)

_known: list[tuple[str, str]] = []  # (value, rule id), longest first
_MIN_KNOWN_LEN = 8


def load_known_secrets(env: dict[str, str] | None = None) -> None:
    """Remember the secret values in Kartrix's own environment (called at import)."""
    found: dict[str, str] = {}
    for name, value in (os.environ if env is None else env).items():
        if not is_secret_var(name, value):
            continue
        if "://" in value:
            password = urlsplit(value).password
            if password and len(password) >= _MIN_KNOWN_LEN:
                found[password] = "env"
        elif len(value) >= _MIN_KNOWN_LEN:
            found[value] = "env"
    _known[:] = sorted(found.items(), key=lambda kv: len(kv[0]), reverse=True)


@dataclass(frozen=True)
class Finding:
    rule: str
    start: int
    end: int


def find_secrets(text: str) -> list[Finding]:
    """Non-overlapping secret spans in ``text`` (known values first, then rules)."""
    spans: list[Finding] = []

    def free(start: int, end: int) -> bool:
        return all(end <= f.start or start >= f.end for f in spans)

    for value, rule_id in _known:
        start = text.find(value)
        while start != -1:
            if free(start, start + len(value)):
                spans.append(Finding(rule_id, start, start + len(value)))
            start = text.find(value, start + len(value))
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            start, end = m.span(rule.group)
            secret = m.group(rule.group)
            if start == end or PLACEHOLDER.fullmatch(secret) or (rule.check and not rule.check(secret)):
                continue
            if free(start, end):
                spans.append(Finding(rule.id, start, end))
    return sorted(spans, key=lambda f: f.start)


def redact_with_count(text: str) -> tuple[str, int]:
    findings = find_secrets(text)
    if not findings:
        return text, 0
    out, pos = [], 0
    for f in findings:
        out += [text[pos : f.start], placeholder(f.rule)]
        pos = f.end
    out.append(text[pos:])
    return "".join(out), len(findings)


def redact(text: str) -> str:
    return redact_with_count(text)[0]


def contains_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER.search(text))


load_known_secrets()
