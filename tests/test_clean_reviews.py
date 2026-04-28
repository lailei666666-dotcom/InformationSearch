from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.preprocess.clean_reviews import clean_reviews_frame
from scripts.preprocess.report_dataset_stats import report_dataset_stats
from src.common.quality import normalize_review_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess" / "clean_reviews.py"


def test_clean_reviews_removes_short_noise_and_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "很好", "category": "键盘"},
            {"review_id": "2", "clean_text": "按键声音很小，图书馆用不会影响别人", "category": "键盘"},
            {"review_id": "3", "clean_text": "按键声音很小，图书馆用不会影响别人", "category": "键盘"},
        ]
    )

    cleaned = clean_reviews_frame(frame)

    assert cleaned["review_id"].tolist() == ["2"]


@pytest.mark.parametrize("missing_column", ["category", "clean_text"])
def test_clean_reviews_requires_category_and_clean_text(missing_column: str) -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "按键反馈清晰，办公使用很舒服", "category": "键盘"},
        ]
    ).drop(columns=[missing_column])

    with pytest.raises(ValueError, match=f"Missing required review columns: {missing_column}"):
        clean_reviews_frame(frame)


def test_clean_reviews_normalizes_category_and_text_before_deduping() -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "按键 反馈清晰", "category": " 键盘 "},
            {"review_id": "2", "clean_text": "按键\n反馈清晰", "category": "键盘"},
            {"review_id": "3", "clean_text": "按键反馈轻盈，晚上用也不吵", "category": "键盘"},
        ]
    )

    cleaned = clean_reviews_frame(frame)

    assert cleaned["review_id"].tolist() == ["1", "3"]
    assert cleaned["clean_text"].tolist()[0] == "按键 反馈清晰"
    assert cleaned["category"].tolist()[0] == "键盘"


def test_clean_reviews_drops_numeric_only_noise() -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "1234567890", "category": "键盘"},
            {"review_id": "2", "clean_text": "键盘回弹很舒服", "category": "键盘"},
        ]
    )

    cleaned = clean_reviews_frame(frame)

    assert cleaned["review_id"].tolist() == ["2"]


def test_report_dataset_stats_normalizes_duplicates_and_category_counts(tmp_path) -> None:
    source = tmp_path / "reviews.csv"
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "按键 反馈清晰", "category": " 键盘 "},
            {"review_id": "2", "clean_text": "按键\n反馈清晰", "category": "键盘"},
            {"review_id": "3", "clean_text": "灯光柔和 适合夜读", "category": "台灯"},
        ]
    )
    frame.to_csv(source, index=False)

    stats = report_dataset_stats(source)
    expected_average = frame["clean_text"].map(normalize_review_text).map(len).mean()

    assert stats["categories"] == {"键盘": 2, "台灯": 1}
    assert stats["duplicate_category_clean_text_rows"] == 1
    assert stats["avg_clean_text_length"] == pytest.approx(expected_average)


def test_clean_reviews_cli_runs_from_command_line(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    output = tmp_path / "cleaned.csv"
    pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "按键回弹清晰，打字很舒服", "category": "键盘"},
            {"review_id": "2", "clean_text": "不错", "category": "键盘"},
        ]
    ).to_csv(source, index=False)

    result = subprocess.run(
        [sys.executable, str(CLEAN_SCRIPT), str(source), str(output)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    cleaned = pd.read_csv(output, dtype={"review_id": str})
    assert cleaned["review_id"].tolist() == ["1"]
