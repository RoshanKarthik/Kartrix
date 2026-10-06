from __future__ import annotations

import hashlib
import os
import time
from urllib.parse import urlsplit

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import array_to_buffer

from kartrix.config import settings
from kartrix.llm.factory import get_embedder
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

VECTOR_DTYPE = "float32"


class CacheConfigError(RuntimeError):
    """Raised when REDIS_URL is missing or not password-protected."""


def get_redis_url() -> str:
    """REDIS_URL from the environment; refuses URLs without a password (B12)."""
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        raise CacheConfigError("REDIS_URL is not set — add it to .env (see .env.example)")
    parts = urlsplit(url)
    if parts.scheme not in ("redis", "rediss"):
        raise CacheConfigError("REDIS_URL must use redis:// or rediss://")
    if not parts.password:
        raise CacheConfigError("REDIS_URL has no password — refusing to use an unauthenticated Redis")
    return url


def _build_index_schema(dims: int, namespace: str) -> dict:
    return {
        # Everything this app writes lives under "<namespace>:" so it can share a Redis
        # with other apps (and be wiped) without touching anything else.
        "index": {"name": f"{namespace}_semantic_cache", "prefix": f"{namespace}:cache:"},
        "fields": [
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "algorithm": "HNSW",
                    "distance_metric": "cosine",
                    "datatype": VECTOR_DTYPE,
                },
            },
            {"name": "response", "type": "text"},
            {"name": "query_text", "type": "text"},
            {"name": "domain", "type": "tag"},
            {"name": "model", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
        ],
    }


class SemanticCache:
    """Async Redis-backed semantic cache for /ask responses.

    Embeddings come from the app's own get_embedder() (so the cache follows
    whichever provider config.yaml selects) and lookups filter on both
    domain and model, since this app can switch LLM providers at runtime
    and a cached answer from one model shouldn't be served under another.
    """

    def __init__(self, redis_url: str, threshold: float, dims: int, namespace: str):
        self.client = Redis.from_url(redis_url)
        self.embedder = get_embedder()
        self.threshold = threshold
        self.prefix = f"{namespace}:cache:"
        self.index = AsyncSearchIndex.from_dict(
            _build_index_schema(dims, namespace), redis_client=self.client
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def setup(self) -> None:
        # overwrite=False: reuse the index if it already exists from a
        # previous run instead of wiping cached entries on every startup.
        await self.index.create(overwrite=False)

    async def _embed(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def _top_match(self, query: str, domain: str, model: str) -> dict | None:
        vector = await self._embed(query)
        q = VectorQuery(
            vector=vector,
            vector_field_name="query_vector",
            return_fields=["response", "query_text", "vector_distance"],
            num_results=1,
            dtype=VECTOR_DTYPE,
        )
        q.set_filter((Tag("domain") == domain) & (Tag("model") == model))
        results = await self.index.query(q)
        return results[0] if results else None

    async def get(self, query: str, domain: str, model: str) -> str | None:
        """Look up a cached response. Returns None on miss."""
        hit = await self._top_match(query, domain=domain, model=model)
        if hit is None:
            return None
        similarity = 1 - float(hit["vector_distance"])
        if similarity >= self.threshold:
            return hit["response"]
        # A candidate exists but isn't close enough - treat as a miss rather
        # than returning a possibly-wrong cached answer.
        return None

    async def put(self, query: str, response: str, domain: str, model: str, ttl: int) -> None:
        """Store a (query, response) pair."""
        vector = await self._embed(query)
        # Key = <ns>:cache:<domain>:<hash(model, query)> — the same question asked in another
        # repo or of another model must not overwrite this entry.
        digest = hashlib.sha256(f"{model}\x00{query}".encode()).hexdigest()[:32]
        redis_key = f"{self.prefix}{domain}:{digest}"
        entry = {
            "query_vector": array_to_buffer(vector, VECTOR_DTYPE),
            "response": response,
            "query_text": query,
            "domain": domain,
            "model": model,
            "created_at": time.time(),
        }
        await self.index.load([entry], keys=[redis_key], ttl=ttl)

    async def invalidate_domain(self, domain: str) -> None:
        """Delete all cache entries for a domain (e.g. after a codebase change).

        The domain is part of the key, so a single prefix scan finds them; ``domain``
        is a hex digest, so it can't smuggle glob characters into the pattern.
        """
        batch: list[bytes] = []
        async for key in self.client.scan_iter(match=f"{self.prefix}{domain}:*", count=500):
            batch.append(key)
            if len(batch) >= 500:
                await self.client.unlink(*batch)
                batch.clear()
        if batch:
            await self.client.unlink(*batch)


def get_repo_domain(repo_path: str) -> str:
    """Stable cache domain derived from the repo path, so cached entries
    persist across restarts for the same repo without colliding with
    entries from a different repo indexed by the same tool."""
    return hashlib.sha256(repo_path.encode()).hexdigest()[:16]


async def build_semantic_cache() -> SemanticCache | None:
    """Build and initialize the semantic cache per config.yaml.

    Returns None (and logs why) if caching is disabled or Redis is
    unreachable, so the caller can run without caching instead of crashing.
    """
    cache_settings = settings.semantic_cache
    if not cache_settings.enabled:
        logger.info("Semantic cache: disabled (semantic_cache.enabled is false in config.yaml)")
        return None

    try:
        dims = settings.embeddings.dims
        cache = SemanticCache(
            redis_url=get_redis_url(),
            threshold=cache_settings.threshold,
            dims=dims,
            namespace=cache_settings.namespace,
        )
        await cache.setup()
        logger.info(f"Semantic cache: enabled (threshold={cache.threshold}, dims={dims})")
        return cache
    except Exception as error:
        logger.warning(f"Semantic cache: disabled (init failed: {error})")
        return None