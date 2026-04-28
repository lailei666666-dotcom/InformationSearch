import pytest

from src.common.retrying import retry


def test_retry_rejects_non_positive_attempts() -> None:
    with pytest.raises(ValueError, match="attempts must be greater than 0"):
        retry(lambda: "ok", attempts=0)


def test_retry_retries_matching_exceptions_until_success() -> None:
    attempts = {"count": 0}

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary timeout")
        return "done"

    result = retry(flaky_operation, attempts=3)

    assert result == "done"
    assert attempts["count"] == 3


def test_retry_does_not_retry_non_matching_exceptions() -> None:
    attempts = {"count": 0}

    def bad_operation() -> str:
        attempts["count"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        retry(bad_operation, attempts=3)

    assert attempts["count"] == 1


def test_retry_accepts_explicit_retryable_exceptions() -> None:
    attempts = {"count": 0}

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("retry me")
        return "done"

    result = retry(flaky_operation, attempts=2, exceptions=(RuntimeError,))

    assert result == "done"
    assert attempts["count"] == 2
