from collections.abc import Callable
from time import sleep
from typing import TypeVar


T = TypeVar("T")
DEFAULT_RETRY_EXCEPTIONS = (OSError, TimeoutError)


def retry(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> T:
    if attempts <= 0:
        raise ValueError("attempts must be greater than 0")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            last_error = exc
            if attempt == attempts:
                break
            if delay_seconds > 0:
                sleep(delay_seconds)
    assert last_error is not None
    raise last_error
