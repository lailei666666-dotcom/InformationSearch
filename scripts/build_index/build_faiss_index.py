from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.io_utils import read_table  # noqa: E402
from src.semantic_retrieval.semantic_engine import SemanticEngine  # noqa: E402

TEXT_COLUMNS = ("clean_text", "raw_text")


def build_faiss_index(source: Path, embeddings: Path, output: Path) -> SemanticEngine:
    frame = read_table(source)
    indexed_frame, vectors, metadata = _align_reviews_with_embeddings(frame, embeddings)
    engine = SemanticEngine.from_vectors(indexed_frame, vectors, metadata=metadata)
    engine.save(output)
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic retrieval index from review rows and vectors.")
    parser.add_argument("source", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_faiss_index(args.source, args.embeddings, args.output)


def _align_reviews_with_embeddings(
    frame: pd.DataFrame,
    embeddings_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    text_column = _select_text_column(frame)
    normalized_text = frame[text_column].map(_normalize_embedding_text)
    candidate_frame = frame.loc[normalized_text.map(bool)].copy()
    candidate_frame[text_column] = normalized_text.loc[candidate_frame.index]
    candidate_frame["review_id"] = candidate_frame["review_id"].astype(str)

    embedding_rows = _load_embedding_rows(embeddings_path)
    embedding_review_ids = [row["review_id"] for row in embedding_rows]

    source_review_ids = set(candidate_frame["review_id"])
    extra_review_ids = sorted(set(embedding_review_ids) - source_review_ids)
    if extra_review_ids:
        raise ValueError(
            "Unknown embeddings for review_id values: " + ", ".join(extra_review_ids)
        )

    by_review_id = {row["review_id"]: row for row in embedding_rows}
    missing_review_ids = sorted(source_review_ids - set(by_review_id))
    if missing_review_ids:
        raise ValueError(
            "Missing embeddings for review_id values: " + ", ".join(missing_review_ids)
        )

    ordered_review_ids = candidate_frame["review_id"].tolist()
    ordered_frame = candidate_frame.reset_index(drop=True)
    vectors = np.asarray([by_review_id[review_id]["vector"] for review_id in ordered_review_ids], dtype="float32")
    metadata = {
        "source_row_count": int(len(frame)),
        "indexed_row_count": int(len(ordered_frame)),
        "vector_count": int(len(vectors)),
        "review_ids": ordered_review_ids,
        "source_text_column": text_column,
        "embedding_source_format": "jsonl",
    }
    return ordered_frame, vectors, metadata


def _load_embedding_rows(source: Path) -> list[dict[str, Any]]:
    if source.suffix.lower() == ".npy":
        raise ValueError("Embedding .npy files are not supported here because review_id alignment is required.")

    rows: list[dict[str, Any]] = []
    seen_review_ids: set[str] = set()
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        review_id = str(record.get("review_id", "")).strip()
        vector = record.get("vector")
        if not review_id:
            raise ValueError("Embedding rows must include a non-empty review_id.")
        if review_id in seen_review_ids:
            raise ValueError(f"Duplicate embeddings found for review_id: {review_id}")
        if not isinstance(vector, list):
            raise ValueError(f"Embedding row for review_id {review_id} is missing a vector list.")
        seen_review_ids.add(review_id)
        rows.append({"review_id": review_id, "vector": [float(value) for value in vector]})
    return rows


def _select_text_column(frame: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError("Missing required review text column: clean_text or raw_text")


def _normalize_embedding_text(value: object) -> str:
    return str(value).strip()


if __name__ == "__main__":
    main()
