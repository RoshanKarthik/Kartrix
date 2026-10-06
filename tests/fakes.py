"""Test doubles."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from kartrix.db.models import EMBEDDING_DIMS

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (hashing trick), no network.

    Texts sharing words get similar vectors, so retrieval tests can assert rankings.
    """

    def __init__(self, dims: int = EMBEDDING_DIMS) -> None:
        self.dims = dims
        self.calls = 0  # number of texts embedded, to assert incremental indexing skips work

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for word in _WORD.findall(text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0  # noqa: S324 — bucketing only
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
