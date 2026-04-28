from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.io_utils import read_table, write_table

RETRIEVAL_COLUMNS = (
    "review_id",
    "product_id",
    "product_name",
    "category",
    "clean_text",
)


def slim_reviews_for_retrieval(source: Path, output: Path) -> pd.DataFrame:
    frame = read_table(source)
    _require_columns(frame, RETRIEVAL_COLUMNS)
    slimmed = frame.loc[:, list(RETRIEVAL_COLUMNS)].copy()
    slimmed = _drop_empty_rows(slimmed)
    write_table(slimmed, output)
    return slimmed


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def _drop_empty_rows(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in RETRIEVAL_COLUMNS:
        cleaned[column] = cleaned[column].map(lambda value: "" if pd.isna(value) else str(value).strip())
    non_empty_mask = cleaned.apply(lambda row: all(row[column] for column in RETRIEVAL_COLUMNS), axis=1)
    return cleaned.loc[non_empty_mask].reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Keep only the minimal retrieval columns needed by BM25, semantic retrieval, and fusion."
    )
    parser.add_argument("source", type=Path, help="Clean review dataset path.")
    parser.add_argument("output", type=Path, help="Slim retrieval dataset path.")
    args = parser.parse_args()
    slim_reviews_for_retrieval(args.source, args.output)


if __name__ == "__main__":
    main()
