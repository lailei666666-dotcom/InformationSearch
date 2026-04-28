from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.common.retrying import retry
from src.semantic_retrieval.embedding_cache import EmbeddingCache

EmbeddingVector = list[float]
EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class EmbeddingClient:
    """Lightweight shared entry point for cached embedding generation."""

    def __init__(
        self,
        *,
        embedder: EmbeddingFunction | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._batch_size = batch_size

    def embed_text(self, text: str) -> EmbeddingVector:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[EmbeddingVector]:
        ordered_texts = list(texts)
        results: list[EmbeddingVector | None] = [None] * len(ordered_texts)
        missing_texts: list[str] = []
        missing_indexes: list[int] = []

        for index, text in enumerate(ordered_texts):
            cached = self._cache.get(text) if self._cache is not None else None
            if cached is not None:
                results[index] = cached
                continue
            missing_texts.append(text)
            missing_indexes.append(index)

        if missing_texts:
            if self._embedder is None:
                raise RuntimeError("Embedding client is not configured with an embedder.")
            for text_batch, index_batch in zip(
                _chunked(missing_texts, self._batch_size),
                _chunked(missing_indexes, self._batch_size),
                strict=True,
            ):
                fresh_vectors = self._embedder(text_batch)
                for index, text, vector in zip(index_batch, text_batch, fresh_vectors, strict=True):
                    normalized = [float(value) for value in vector]
                    if self._cache is not None:
                        self._cache.set(text, normalized)
                    results[index] = normalized

        return [vector for vector in results if vector is not None]


def build_openai_compatible_embedder(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    dimensions: int | None = None,
    timeout_seconds: float = 30.0,
) -> EmbeddingFunction:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate uncached embeddings.")

    endpoint = base_url.rstrip("/") + "/embeddings"

    def _embed(texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": model, "input": list(texts)}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        request_body = json.dumps(payload).encode("utf-8")

        def _request() -> list[list[float]]:
            request = Request(
                endpoint,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.load(response)
            except HTTPError as exc:
                if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                    raise URLError(f"Transient embedding HTTP error {exc.code}") from exc
                raise
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object from embedding endpoint")

            data = payload.get("data")
            if not isinstance(data, list):
                raise ValueError("Embedding response is missing a data list")

            vectors: list[list[float]] = []
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("Embedding response items must be JSON objects")
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    raise ValueError("Embedding response items must include an embedding list")
                vectors.append([float(value) for value in embedding])

            if len(vectors) != len(texts):
                raise ValueError("Embedding response count does not match request count")
            return vectors

        return retry(_request, attempts=6, delay_seconds=2.0, exceptions=(URLError, TimeoutError))

    return _embed


def _chunked(values: Sequence[str] | Sequence[int], batch_size: int | None) -> list[Sequence[str] | Sequence[int]]:
    if batch_size is None or batch_size <= 0:
        return [values]
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]
