from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.common.retrying import retry


@dataclass(slots=True)
class HttpClient:
    timeout_seconds: float = 10.0
    user_agent: str = "InformationSearch/0.1"

    def get_json(self, url: str) -> dict[str, Any]:
        def _request() -> dict[str, Any]:
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return payload

        return retry(_request, exceptions=(URLError, TimeoutError))
