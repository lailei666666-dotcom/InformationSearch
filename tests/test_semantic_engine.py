from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_index.build_faiss_index import build_faiss_index
from src.semantic_retrieval.faiss_index import FaissIndex
from src.semantic_retrieval.semantic_engine import SemanticEngine


def test_semantic_engine_filters_by_category_before_ranking() -> None:
    frame = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "K1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "quiet keys for the library",
            },
            {
                "review_id": "R2",
                "product_id": "L1",
                "product_name": "Lamp A",
                "category": "lamp",
                "clean_text": "soft light at night",
            },
        ]
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    engine = SemanticEngine.from_vectors(frame, vectors)
    results = engine.search(
        query_vector=np.array([1.0, 0.0], dtype="float32"),
        category="keyboard",
        top_k=2,
    )

    assert results[0]["review_id"] == "R1"
    assert all(item["category"] == "keyboard" for item in results)


def test_semantic_engine_returns_ranked_scores() -> None:
    frame = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "light touch typing",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "Keyboard B",
                "category": "keyboard",
                "clean_text": "crisp feedback",
            },
        ]
    )
    vectors = np.array([[1.0, 0.0], [0.6, 0.8]], dtype="float32")

    engine = SemanticEngine.from_vectors(frame, vectors)
    results = engine.search(
        query_vector=np.array([1.0, 0.0], dtype="float32"),
        top_k=2,
    )

    assert [item["review_id"] for item in results] == ["R1", "R2"]
    assert [item["rank"] for item in results] == [1, 2]
    assert results[0]["score"] >= results[1]["score"]


def test_semantic_engine_save_load_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "silent typing",
            }
        ]
    )
    vectors = np.array([[3.0, 4.0]], dtype="float32")
    engine = SemanticEngine.from_vectors(frame, vectors)

    output = tmp_path / "semantic-index.json"
    engine.save(output)
    reloaded = SemanticEngine.load(output)

    assert reloaded.documents.to_dict(orient="records") == engine.documents.to_dict(orient="records")
    np.testing.assert_allclose(reloaded.index.vectors, engine.index.vectors)
    assert reloaded.metadata["review_ids"] == ["R1"]


def test_semantic_engine_load_validates_provenance_metadata(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "silent typing",
            }
        ]
    )
    engine = SemanticEngine.from_vectors(
        frame,
        np.array([[1.0, 0.0]], dtype="float32"),
        metadata={
            "source_row_count": 1,
            "source_text_column": "clean_text",
            "embedding_source_format": "jsonl",
        },
    )
    output = tmp_path / "semantic-index.json"
    engine.save(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["metadata"]["embedding_source_format"] = "npy"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="embedding_source_format"):
        SemanticEngine.load(output)


def test_build_faiss_index_aligns_embeddings_by_review_id(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "silent keys",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "Lamp A",
                "category": "lamp",
                "clean_text": "   ",
            },
            {
                "review_id": "R3",
                "product_id": "P3",
                "product_name": "Keyboard B",
                "category": "keyboard",
                "clean_text": "great travel board",
            },
        ]
    ).to_csv(source, index=False)

    embeddings = tmp_path / "embeddings.jsonl"
    embeddings.write_text(
        "\n".join(
            [
                json.dumps({"review_id": "R3", "text": "great travel board", "vector": [0.0, 1.0]}),
                json.dumps({"review_id": "R1", "text": "silent keys", "vector": [1.0, 0.0]}),
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "semantic-index.json"
    engine = build_faiss_index(source, embeddings, output)

    assert engine.documents["review_id"].tolist() == ["R1", "R3"]
    np.testing.assert_allclose(
        engine.index.vectors,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    )
    assert engine.metadata["source_row_count"] == 3
    assert engine.metadata["indexed_row_count"] == 2


def test_build_faiss_index_matches_embed_reviews_text_normalization(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": 12345,
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "Keyboard B",
                "category": "keyboard",
                "clean_text": "   ",
            },
        ]
    ).to_csv(source, index=False)

    embeddings = tmp_path / "embeddings.jsonl"
    embeddings.write_text(
        json.dumps({"review_id": "R1", "text": "12345", "vector": [1.0, 0.0]}),
        encoding="utf-8",
    )

    engine = build_faiss_index(source, embeddings, tmp_path / "semantic-index.json")

    assert engine.documents["review_id"].tolist() == ["R1"]
    assert engine.documents.iloc[0]["clean_text"] == "12345"


def test_build_faiss_index_fails_on_missing_embedding_rows(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "silent keys",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "Keyboard B",
                "category": "keyboard",
                "clean_text": "clicky switches",
            },
        ]
    ).to_csv(source, index=False)

    embeddings = tmp_path / "embeddings.jsonl"
    embeddings.write_text(
        json.dumps({"review_id": "R1", "text": "silent keys", "vector": [1.0, 0.0]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing embeddings for review_id values: R2"):
        build_faiss_index(source, embeddings, tmp_path / "semantic-index.json")


def test_build_faiss_index_fails_on_unknown_embedding_rows(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Keyboard A",
                "category": "keyboard",
                "clean_text": "silent keys",
            }
        ]
    ).to_csv(source, index=False)

    embeddings = tmp_path / "embeddings.jsonl"
    embeddings.write_text(
        "\n".join(
            [
                json.dumps({"review_id": "R1", "text": "silent keys", "vector": [1.0, 0.0]}),
                json.dumps({"review_id": "R9", "text": "ghost row", "vector": [0.0, 1.0]}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown embeddings for review_id values: R9"):
        build_faiss_index(source, embeddings, tmp_path / "semantic-index.json")


@pytest.mark.parametrize(
    ("allowed_indices", "message"),
    [
        (np.array([-1], dtype=np.int64), "must not contain negative"),
        (np.array([2], dtype=np.int64), "must be smaller than vector count"),
        (np.array([0, 0], dtype=np.int64), "must not contain duplicates"),
    ],
)
def test_faiss_index_validates_allowed_indices(
    allowed_indices: np.ndarray,
    message: str,
) -> None:
    index = FaissIndex.from_vectors(np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))

    with pytest.raises(ValueError, match=message):
        index.search(
            np.array([1.0, 0.0], dtype="float32"),
            top_k=1,
            allowed_indices=allowed_indices,
        )
