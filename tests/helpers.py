"""Helpers shared by fixtures and tests."""

import os

import pytest

REQUIRE_SERVICES = os.environ.get("KARTRIX_REQUIRE_SERVICES") == "1"


def unavailable(reason: str) -> None:
    """Skip a test whose service is down — or fail it in CI (KARTRIX_REQUIRE_SERVICES=1)."""
    if REQUIRE_SERVICES:
        pytest.fail(reason)
    pytest.skip(reason)
