from collections.abc import AsyncIterator

import pytest

from kartrix.cache.semantic_cache import CacheConfigError, SemanticCache, get_redis_url, get_repo_domain
from tests.fakes import HashingEmbeddings
from tests.helpers import unavailable

NS = "kartrix_test"


@pytest.mark.parametrize(
    "url",
    ["", "redis://localhost:6379/0", "redis://user@localhost:6379", "http://:pw@localhost:6379"],
)
def test_unauthenticated_or_odd_urls_are_refused(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("REDIS_URL", url)
    with pytest.raises(CacheConfigError):
        get_redis_url()


@pytest.fixture
async def cache(redis_url: str, fake_embedder: HashingEmbeddings) -> AsyncIterator[SemanticCache]:
    c = SemanticCache(redis_url=redis_url, threshold=0.85, dims=2048, namespace=NS)
    try:
        await c.setup()
    except Exception as e:
        unavailable(f"Redis unavailable: {e}")
    yield c
    async for key in c.client.scan_iter(match=f"{NS}:*"):
        await c.client.unlink(key)
    await c.index.delete(drop=True)
    await c.aclose()


async def test_keys_are_namespaced_and_isolated(cache: SemanticCache) -> None:
    a, b = get_repo_domain("D:/repoA"), get_repo_domain("D:/repoB")
    q = "what does index_repo do?"
    await cache.put(q, "A-m1", domain=a, model="m1", ttl=60)
    await cache.put(q, "B-m1", domain=b, model="m1", ttl=60)
    await cache.put(q, "A-m2", domain=a, model="m2", ttl=60)  # used to overwrite A-m1

    keys = [k.decode() async for k in cache.client.scan_iter(match=f"{NS}:*")]
    assert len(keys) == 3 and all(k.startswith(f"{NS}:cache:") for k in keys)
    assert all(0 < ttl <= 60 for ttl in [await cache.client.ttl(k) for k in keys])

    assert await cache.get(q, a, "m1") == "A-m1"
    assert await cache.get(q, a, "m2") == "A-m2"
    assert await cache.get(q, b, "m1") == "B-m1"
    assert await cache.get("how do I bake sourdough bread", a, "m1") is None


async def test_invalidate_only_touches_one_repo(cache: SemanticCache) -> None:
    a, b = get_repo_domain("D:/repoA"), get_repo_domain("D:/repoB")
    await cache.put("question one", "A", domain=a, model="m", ttl=60)
    await cache.put("question one", "B", domain=b, model="m", ttl=60)
    await cache.invalidate_domain(a)
    assert await cache.get("question one", a, "m") is None
    assert await cache.get("question one", b, "m") == "B"
