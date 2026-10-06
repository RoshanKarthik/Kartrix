import httpx
import pytest

from kartrix.config import RetrySettings
from kartrix.llm.retry import acall_with_retry, call_with_retry, is_transient, status_code_of

FAST = RetrySettings(max_retries=3, initial_delay=0.001, backoff_factor=1.0, max_delay=0.001)


class StatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.mark.parametrize(("status", "transient"), [(429, True), (503, True), (408, True), (401, False), (404, False)])
def test_status_classification(status: int, transient: bool) -> None:
    assert is_transient(StatusError(status)) is transient


def test_nvidia_message_prefix_is_parsed() -> None:
    assert status_code_of(Exception("[502] Bad Gateway from upstream")) == 502


def test_connection_errors_are_transient() -> None:
    assert is_transient(httpx.ConnectError("refused"))


def test_retries_transient_then_succeeds() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise StatusError(503)
        return "ok"

    assert call_with_retry(fn, FAST, "test") == "ok"
    assert len(calls) == 3


def test_permanent_error_is_not_retried() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        raise StatusError(401)

    with pytest.raises(StatusError):
        call_with_retry(fn, FAST, "test")
    assert len(calls) == 1


async def test_async_gives_up_after_max_retries() -> None:
    calls = []

    async def fn() -> str:
        calls.append(1)
        raise StatusError(429)

    with pytest.raises(StatusError):
        await acall_with_retry(fn, FAST, "test")
    assert len(calls) == FAST.max_retries + 1
