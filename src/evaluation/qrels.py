from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

QUERY_REQUIRED_COLUMNS = ("query_id", "query_type", "category", "query_text")
REVIEW_QRELS_REQUIRED_COLUMNS = ("query_id", "review_id", "relevance", "category")
PRODUCT_QRELS_REQUIRED_COLUMNS = ("query_id", "product_id", "relevance", "category")


def load_query_set(annotations_dir: Path | None = None) -> pd.DataFrame:
    annotations_root = _resolve_annotations_dir(annotations_dir)
    frame = _read_annotation_csv(annotations_root / "queries.csv")
    _require_columns(frame, QUERY_REQUIRED_COLUMNS, "queries.csv")
    _reject_duplicate_rows(frame, ("query_id",), "queries.csv", "Duplicate query_id rows found")
    return frame


def load_review_qrels(annotations_dir: Path | None = None) -> pd.DataFrame:
    annotations_root = _resolve_annotations_dir(annotations_dir)
    query_set = load_query_set(annotations_root)
    frame = _read_annotation_csv(annotations_root / "qrels_reviews.csv")
    _require_columns(frame, REVIEW_QRELS_REQUIRED_COLUMNS, "qrels_reviews.csv")
    _reject_duplicate_rows(
        frame,
        ("query_id", "review_id"),
        "qrels_reviews.csv",
        "Duplicate qrels_reviews.csv pairs found for columns: query_id, review_id",
    )
    _validate_known_query_ids(frame, query_set, "qrels_reviews.csv")
    _validate_matching_categories(frame, query_set, "qrels_reviews.csv")
    return frame


def load_product_qrels(annotations_dir: Path | None = None) -> pd.DataFrame:
    annotations_root = _resolve_annotations_dir(annotations_dir)
    query_set = load_query_set(annotations_root)
    frame = _read_annotation_csv(annotations_root / "qrels_products.csv")
    _require_columns(frame, PRODUCT_QRELS_REQUIRED_COLUMNS, "qrels_products.csv")
    _reject_duplicate_rows(
        frame,
        ("query_id", "product_id"),
        "qrels_products.csv",
        "Duplicate qrels_products.csv pairs found for columns: query_id, product_id",
    )
    _validate_known_query_ids(frame, query_set, "qrels_products.csv")
    _validate_matching_categories(frame, query_set, "qrels_products.csv")
    return frame


def _resolve_annotations_dir(annotations_dir: Path | None) -> Path:
    return ANNOTATIONS_DIR if annotations_dir is None else annotations_dir


def _read_annotation_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...], filename: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(missing)
        raise ValueError(f"Missing required columns in {filename}: {missing_display}")


def _reject_duplicate_rows(
    frame: pd.DataFrame,
    subset: tuple[str, ...],
    filename: str,
    message: str,
) -> None:
    del filename
    if frame.duplicated(subset=list(subset), keep=False).any():
        raise ValueError(message)


def _validate_known_query_ids(frame: pd.DataFrame, query_set: pd.DataFrame, filename: str) -> None:
    known_query_ids = set(query_set["query_id"])
    unknown_query_ids = sorted(set(frame["query_id"]) - known_query_ids)
    if unknown_query_ids:
        unknown_display = ", ".join(unknown_query_ids)
        raise ValueError(f"Unknown query_id values in {filename}: {unknown_display}")


def _validate_matching_categories(frame: pd.DataFrame, query_set: pd.DataFrame, filename: str) -> None:
    query_categories = query_set.loc[:, ["query_id", "category"]].rename(columns={"category": "query_category"})
    merged = frame.merge(query_categories, on="query_id", how="left")
    mismatched_query_ids = sorted(
        merged.loc[merged["category"] != merged["query_category"], "query_id"].drop_duplicates().tolist()
    )
    if mismatched_query_ids:
        mismatched_display = ", ".join(mismatched_query_ids)
        raise ValueError(f"Category mismatch in {filename} for query_ids: {mismatched_display}")
