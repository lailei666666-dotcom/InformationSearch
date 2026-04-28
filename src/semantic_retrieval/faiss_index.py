from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only when faiss is absent
    faiss = None


@dataclass(slots=True)
class FaissIndex:
    """Small exact-vector index with FAISS when available and NumPy as fallback."""

    vectors: np.ndarray
    dimension: int = field(init=False)
    _index: Any | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = _normalize_vectors(self.vectors)
        self.vectors = normalized
        self.dimension = int(normalized.shape[1]) if len(normalized) else 0
        self._index = _build_faiss_index(normalized) if self.dimension else None

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> "FaissIndex":
        return cls(vectors=np.asarray(vectors, dtype="float32"))

    @classmethod
    def load(cls, source: Path) -> "FaissIndex":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_vectors(np.asarray(payload["vectors"], dtype="float32"))

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vectors": self.vectors.tolist()}
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        allowed_indices: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or self.dimension == 0:
            return []

        query = _normalize_query(np.asarray(query_vector, dtype="float32"), self.dimension)
        if allowed_indices is None and self._index is not None:
            return self._search_full_index(query, top_k)

        candidate_indices = _normalize_allowed_indices(allowed_indices, len(self.vectors))
        if candidate_indices.size == 0:
            return []

        scores = self.vectors[candidate_indices] @ query
        limited_top_k = min(top_k, int(candidate_indices.size))
        order = np.argsort(-scores, kind="stable")[:limited_top_k]

        results: list[dict[str, Any]] = []
        for rank, position in enumerate(order, start=1):
            index = int(candidate_indices[position])
            results.append(
                {
                    "index": index,
                    "score": float(scores[position]),
                    "rank": rank,
                }
            )
        return results

    def _search_full_index(self, query: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        distances, indices = self._index.search(query.reshape(1, -1), min(top_k, len(self.vectors)))
        results: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(zip(distances[0], indices[0], strict=True), start=1):
            if int(index) < 0:
                continue
            results.append({"index": int(index), "score": float(score), "rank": rank})
        return results


def _build_faiss_index(vectors: np.ndarray) -> Any | None:
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("Vectors must be a 2D matrix.")
    normalized = vectors.astype("float32", copy=True)
    if normalized.size == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return normalized / norms


def _normalize_query(query_vector: np.ndarray, dimension: int) -> np.ndarray:
    if query_vector.ndim != 1:
        raise ValueError("Query vector must be 1D.")
    if query_vector.shape[0] != dimension:
        raise ValueError("Query vector dimension does not match the index.")

    normalized = query_vector.astype("float32", copy=True)
    norm = float(np.linalg.norm(normalized))
    if norm == 0.0:
        raise ValueError("Query vector must be non-zero.")
    return normalized / norm


def _normalize_allowed_indices(
    allowed_indices: np.ndarray | None,
    vector_count: int,
) -> np.ndarray:
    if allowed_indices is None:
        return np.arange(vector_count, dtype=np.int64)
    normalized = np.asarray(allowed_indices, dtype=np.int64)
    if normalized.ndim != 1:
        raise ValueError("Allowed indices must be 1D.")
    if np.any(normalized < 0):
        raise ValueError("Allowed indices must not contain negative values.")
    if np.any(normalized >= vector_count):
        raise ValueError("Allowed indices must be smaller than vector count.")
    if len(np.unique(normalized)) != len(normalized):
        raise ValueError("Allowed indices must not contain duplicates.")
    return normalized
