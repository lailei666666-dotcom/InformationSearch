from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.semantic_retrieval.faiss_index import FaissIndex

REQUIRED_COLUMNS = ("review_id", "product_id", "product_name", "category", "clean_text")
TEXT_COLUMNS = ("clean_text", "raw_text")
EMBEDDING_SOURCE_FORMATS = {"jsonl"}


@dataclass(slots=True)
class SemanticEngine:
    documents: pd.DataFrame
    index: FaissIndex
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_vectors(
        cls,
        frame: pd.DataFrame,
        vectors: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "SemanticEngine":
        _require_columns(frame, REQUIRED_COLUMNS)
        documents = frame.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True).copy()
        if len(documents) != len(vectors):
            raise ValueError("Review rows and vectors must have the same length.")
        combined_metadata = _build_default_metadata(documents)
        if metadata:
            combined_metadata.update(metadata)
        _validate_metadata(combined_metadata, documents, len(vectors))
        return cls(
            documents=documents,
            index=FaissIndex.from_vectors(vectors),
            metadata=combined_metadata,
        )

    @classmethod
    def load(cls, source: Path) -> "SemanticEngine":
        payload = json.loads(source.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["documents"], columns=list(REQUIRED_COLUMNS))
        vectors = np.asarray(payload["vectors"], dtype="float32")
        return cls.from_vectors(frame, vectors, metadata=payload.get("metadata"))

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents.to_dict(orient="records"),
            "vectors": self.index.vectors.tolist(),
            "metadata": self.metadata,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(
        self,
        *,
        query_vector: np.ndarray,
        category: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        allowed_indices = _filter_indices(self.documents, category)
        hits = self.index.search(
            np.asarray(query_vector, dtype="float32"),
            top_k=top_k,
            allowed_indices=allowed_indices,
        )

        results: list[dict[str, Any]] = []
        for hit in hits:
            row = self.documents.iloc[hit["index"]].to_dict()
            row["score"] = hit["score"]
            row["rank"] = hit["rank"]
            results.append(row)
        return results


def _filter_indices(frame: pd.DataFrame, category: str | None) -> np.ndarray | None:
    if category is None:
        return None
    normalized = category.strip()
    if not normalized:
        return None
    mask = frame["category"].astype(str) == normalized
    return np.flatnonzero(mask.to_numpy())


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def _build_default_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_ids": frame["review_id"].astype(str).tolist(),
        "indexed_row_count": int(len(frame)),
        "vector_count": int(len(frame)),
    }


def _validate_metadata(
    metadata: dict[str, Any],
    frame: pd.DataFrame,
    vector_count: int,
) -> None:
    review_ids = frame["review_id"].astype(str).tolist()
    indexed_row_count = len(frame)

    if "review_ids" in metadata and metadata["review_ids"] != review_ids:
        raise ValueError("Semantic index metadata review_ids do not match serialized documents.")
    if "indexed_row_count" in metadata and int(metadata["indexed_row_count"]) != indexed_row_count:
        raise ValueError("Semantic index metadata indexed_row_count does not match serialized documents.")
    if "vector_count" in metadata and int(metadata["vector_count"]) != vector_count:
        raise ValueError("Semantic index metadata vector_count does not match serialized vectors.")
    if "schema_version" in metadata and int(metadata["schema_version"]) != 1:
        raise ValueError("Semantic index metadata schema_version is not supported.")
    if "source_row_count" in metadata and int(metadata["source_row_count"]) < indexed_row_count:
        raise ValueError("Semantic index metadata source_row_count cannot be smaller than indexed_row_count.")
    if "source_text_column" in metadata and metadata["source_text_column"] not in TEXT_COLUMNS:
        raise ValueError("Semantic index metadata source_text_column is invalid.")
    if (
        "embedding_source_format" in metadata
        and metadata["embedding_source_format"] not in EMBEDDING_SOURCE_FORMATS
    ):
        raise ValueError("Semantic index metadata embedding_source_format is invalid.")
