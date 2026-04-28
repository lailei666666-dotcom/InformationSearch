from __future__ import annotations

from urllib.error import HTTPError

from src.semantic_retrieval.embedding_client import build_openai_compatible_embedder


def test_openai_compatible_embedder_retries_transient_http_errors(monkeypatch) -> None:
    attempts = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"data":[{"embedding":[0.1,0.2,0.3]}]}'

    def fake_urlopen(request, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)
        return FakeResponse()

    monkeypatch.setattr("src.semantic_retrieval.embedding_client.urlopen", fake_urlopen)
    embedder = build_openai_compatible_embedder(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        dimensions=3,
    )

    assert embedder(["quiet keyboard"]) == [[0.1, 0.2, 0.3]]
    assert attempts["count"] == 2
