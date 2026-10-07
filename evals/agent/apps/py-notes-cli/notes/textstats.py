"""Statistics over note texts."""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9']+")


def word_count(text: str) -> int:
    """Number of words in ``text`` (words are separated by any whitespace)."""
    return len(text.split())


def top_words(texts: list[str], n: int = 3) -> list[tuple[str, int]]:
    """The ``n`` most common words (lower-cased) across ``texts``, most common first."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_WORD.findall(text.lower()))
    return counts.most_common(n)
