from pathlib import Path

from src.common.dedupe import duplicate_mask
from src.common.io_utils import read_table
from src.common.quality import low_quality_text_mask
from src.common.quality import normalize_review_text


def report_dataset_stats(source: Path) -> dict[str, object]:
    frame = read_table(source)
    normalized = frame.copy()
    has_clean_text = "clean_text" in frame
    has_category = "category" in frame
    if has_clean_text:
        normalized["clean_text"] = normalized["clean_text"].map(normalize_review_text)
    if has_category:
        normalized["category"] = normalized["category"].map(normalize_review_text)
    duplicate_rows = (
        int(duplicate_mask(normalized, ("category", "clean_text")).sum())
        if has_clean_text and has_category
        else 0
    )
    low_quality_rows = int(low_quality_text_mask(normalized["clean_text"]).sum()) if has_clean_text else 0
    stats = {
        "rows": int(len(frame)),
        "categories": normalized["category"].value_counts().to_dict() if has_category else {},
        "avg_clean_text_length": (
            float(normalized["clean_text"].fillna("").map(len).mean()) if has_clean_text else 0.0
        ),
        "duplicate_category_clean_text_rows": duplicate_rows,
        "low_quality_clean_text_rows": low_quality_rows,
    }
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report basic statistics for normalized review data.")
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for key, value in report_dataset_stats(args.source).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
