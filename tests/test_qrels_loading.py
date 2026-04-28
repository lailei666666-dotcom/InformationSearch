from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess.bootstrap_annotations import bootstrap_annotation_assets
from scripts.preprocess.clean_reviews import clean_reviews
from scripts.preprocess.slim_reviews_for_retrieval import slim_reviews_for_retrieval
from src.evaluation import qrels as qrels_module
from src.evaluation.qrels import (
    load_product_qrels,
    load_query_set,
    load_review_qrels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_query_set_covers_three_query_types() -> None:
    query_set = load_query_set()

    assert {"关键词明确型", "体验反馈型", "场景化隐含需求型"} == set(query_set["query_type"].unique())


def test_load_query_set_rejects_duplicate_query_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "queries.csv",
        pd.DataFrame(
            [
                {"query_id": "Q001", "query_type": "关键词明确型", "category": "键盘", "query_text": "静音机械键盘"},
                {"query_id": "Q001", "query_type": "体验反馈型", "category": "键盘", "query_text": "键盘久打累不累"},
            ]
        ),
    )
    _write_csv(
        tmp_path / "qrels_reviews.csv",
        pd.DataFrame([{"query_id": "Q001", "review_id": "R1", "relevance": 2, "category": "键盘"}]),
    )
    _write_csv(
        tmp_path / "qrels_products.csv",
        pd.DataFrame([{"query_id": "Q001", "product_id": "P1", "relevance": 3, "category": "键盘"}]),
    )
    monkeypatch.setattr(qrels_module, "ANNOTATIONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Duplicate query_id rows found"):
        load_query_set()


def test_load_product_qrels_requires_expected_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_valid_queries(tmp_path / "queries.csv")
    _write_csv(
        tmp_path / "qrels_reviews.csv",
        pd.DataFrame([{"query_id": "Q001", "review_id": "R1", "relevance": 2, "category": "键盘"}]),
    )
    _write_csv(
        tmp_path / "qrels_products.csv",
        pd.DataFrame([{"query_id": "Q001", "relevance": 3, "category": "键盘"}]),
    )
    monkeypatch.setattr(qrels_module, "ANNOTATIONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Missing required columns in qrels_products.csv: product_id"):
        load_product_qrels()


def test_load_review_qrels_rejects_unknown_query_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_valid_queries(tmp_path / "queries.csv")
    _write_csv(
        tmp_path / "qrels_reviews.csv",
        pd.DataFrame([{"query_id": "Q999", "review_id": "R1", "relevance": 2, "category": "键盘"}]),
    )
    _write_csv(
        tmp_path / "qrels_products.csv",
        pd.DataFrame([{"query_id": "Q001", "product_id": "P1", "relevance": 3, "category": "键盘"}]),
    )
    monkeypatch.setattr(qrels_module, "ANNOTATIONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Unknown query_id values in qrels_reviews.csv: Q999"):
        load_review_qrels()


def test_load_product_qrels_rejects_duplicate_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_valid_queries(tmp_path / "queries.csv")
    _write_csv(
        tmp_path / "qrels_reviews.csv",
        pd.DataFrame([{"query_id": "Q001", "review_id": "R1", "relevance": 2, "category": "键盘"}]),
    )
    _write_csv(
        tmp_path / "qrels_products.csv",
        pd.DataFrame(
            [
                {"query_id": "Q001", "product_id": "P1", "relevance": 3, "category": "键盘"},
                {"query_id": "Q001", "product_id": "P1", "relevance": 2, "category": "键盘"},
            ]
        ),
    )
    monkeypatch.setattr(qrels_module, "ANNOTATIONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Duplicate qrels_products.csv pairs found for columns: query_id, product_id"):
        load_product_qrels()


def test_load_review_qrels_rejects_category_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_valid_queries(tmp_path / "queries.csv")
    _write_csv(
        tmp_path / "qrels_reviews.csv",
        pd.DataFrame([{"query_id": "Q001", "review_id": "R1", "relevance": 2, "category": "台灯"}]),
    )
    _write_csv(
        tmp_path / "qrels_products.csv",
        pd.DataFrame([{"query_id": "Q001", "product_id": "P1", "relevance": 3, "category": "键盘"}]),
    )
    monkeypatch.setattr(qrels_module, "ANNOTATIONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Category mismatch in qrels_reviews.csv for query_ids: Q001"):
        load_review_qrels()


def test_bootstrap_annotation_assets_force_copies_checked_in_assets(tmp_path: Path) -> None:
    output_dir = tmp_path / "annotations"
    source_queries = qrels_module.ANNOTATIONS_DIR / "queries.csv"
    source_review_qrels = qrels_module.ANNOTATIONS_DIR / "qrels_reviews.csv"
    source_product_qrels = qrels_module.ANNOTATIONS_DIR / "qrels_products.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "queries.csv").write_text("query_id\nstale\n", encoding="utf-8")
    (output_dir / "qrels_reviews.csv").write_text("query_id\nstale\n", encoding="utf-8")
    (output_dir / "qrels_products.csv").write_text("query_id\nstale\n", encoding="utf-8")

    written_paths = bootstrap_annotation_assets(output_dir=output_dir, force=True)

    assert written_paths == [
        output_dir / "queries.csv",
        output_dir / "qrels_reviews.csv",
        output_dir / "qrels_products.csv",
    ]
    assert (output_dir / "queries.csv").read_text(encoding="utf-8") == source_queries.read_text(encoding="utf-8")
    assert (output_dir / "qrels_reviews.csv").read_text(encoding="utf-8") == source_review_qrels.read_text(encoding="utf-8")
    assert (output_dir / "qrels_products.csv").read_text(encoding="utf-8") == source_product_qrels.read_text(encoding="utf-8")


def test_checked_in_qrels_align_with_regenerated_xiaomi_retrieval_dataset(tmp_path: Path) -> None:
    clean_path = tmp_path / "xiaomi_reviews_clean.csv"
    dataset_path = tmp_path / "xiaomi_reviews_retrieval.csv"
    clean_reviews(PROJECT_ROOT / "data" / "raw" / "xiaomi_reviews.csv", clean_path)
    slim_reviews_for_retrieval(clean_path, dataset_path)

    dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    review_qrels = load_review_qrels()
    product_qrels = load_product_qrels()

    known_review_ids = set(dataset["review_id"].astype(str))
    known_product_ids = set(dataset["product_id"].astype(str))
    missing_review_ids = sorted(set(review_qrels["review_id"].astype(str)) - known_review_ids)
    missing_product_ids = sorted(set(product_qrels["product_id"].astype(str)) - known_product_ids)

    assert not missing_review_ids, missing_review_ids
    assert not missing_product_ids, missing_product_ids


def _write_valid_queries(path: Path) -> None:
    _write_csv(
        path,
        pd.DataFrame(
            [
                {"query_id": "Q001", "query_type": "关键词明确型", "category": "键盘", "query_text": "静音机械键盘"},
                {"query_id": "Q002", "query_type": "体验反馈型", "category": "台灯", "query_text": "台灯久看刺眼吗"},
            ]
        ),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.write_text(frame.to_csv(index=False), encoding="utf-8")
