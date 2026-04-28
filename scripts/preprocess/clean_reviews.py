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

from src.common.dedupe import drop_duplicates
from src.common.io_utils import read_table, write_table
from src.common.quality import low_quality_text_mask, normalize_review_text

DEDUPLICATION_COLUMNS = ("category", "clean_text")
REQUIRED_COLUMNS = DEDUPLICATION_COLUMNS


def clean_reviews_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_COLUMNS)

    cleaned = frame.copy()
    cleaned["clean_text"] = cleaned["clean_text"].map(normalize_review_text)
    cleaned["category"] = cleaned["category"].map(normalize_review_text)

    quality_mask = ~low_quality_text_mask(cleaned["clean_text"])
    cleaned = cleaned.loc[quality_mask].copy()
    cleaned = drop_duplicates(cleaned, DEDUPLICATION_COLUMNS)
    return cleaned.reset_index(drop=True)


def clean_reviews(source: Path, output: Path) -> pd.DataFrame:
    frame = read_table(source)
    cleaned = clean_reviews_frame(frame)
    write_table(cleaned, output)
    return cleaned


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Filter noisy reviews and remove exact duplicates.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    clean_reviews(args.source, args.output)


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


if __name__ == "__main__":
    main()
