from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.common.config import load_settings
from src.semantic_retrieval.embedding_cache import EmbeddingCache
from src.semantic_retrieval.embedding_client import EmbeddingClient, build_openai_compatible_embedder
from src.semantic_retrieval.local_encoder import LocalSemanticEncoder

DEFAULT_EMBEDDING_CACHE_PATH = Path("outputs") / "semantic" / "embedding-cache.jsonl"


def build_default_embedding_client(cache_path: Path | None = None) -> EmbeddingClient:
    settings = load_settings()
    if settings.embedding.provider != "openai_compatible":
        raise ValueError(
            f"Unsupported embedding provider for default client: {settings.embedding.provider}"
        )

    resolved_cache_path = cache_path or DEFAULT_EMBEDDING_CACHE_PATH
    cache = EmbeddingCache(
        resolved_cache_path,
        namespace=cache_namespace(
            settings.embedding.provider,
            settings.embedding.model,
            settings.embedding.dimensions,
        ),
    )
    embedder = build_openai_compatible_embedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=settings.embedding.model,
        dimensions=settings.embedding.dimensions,
    )
    return EmbeddingClient(embedder=embedder, cache=cache, batch_size=settings.embedding.batch_size)


def encode_query_text(
    query_text: str,
    *,
    encoder_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> np.ndarray:
    if (encoder_path is None) == (embedding_client is None):
        raise ValueError("Provide exactly one of encoder_path or embedding_client.")
    if encoder_path is not None:
        encoder = LocalSemanticEncoder.load(encoder_path)
        return encoder.encode_text(query_text)
    assert embedding_client is not None
    return np.asarray(embedding_client.embed_text(query_text), dtype="float32")


def cache_namespace(provider: str, model: str, dimensions: int | None) -> str:
    dimensions_value = "default" if dimensions is None else str(dimensions)
    return f"{provider}:{model}:{dimensions_value}"
