from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.preprocess.slim_reviews_for_retrieval import slim_reviews_for_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIM_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess" / "slim_reviews_for_retrieval.py"


def test_slim_reviews_for_retrieval_keeps_only_required_columns(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    output = tmp_path / "retrieval.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "键盘A",
                "category": "键盘",
                "clean_text": "手感清晰",
                "rating": 5,
                "review_time": "2026-04-27",
            }
        ]
    ).to_csv(source, index=False)

    frame = slim_reviews_for_retrieval(source, output)

    assert frame.columns.tolist() == [
        "review_id",
        "product_id",
        "product_name",
        "category",
        "clean_text",
    ]
    persisted = pd.read_csv(output, encoding="utf-8-sig")
    assert persisted.columns.tolist() == frame.columns.tolist()


def test_slim_reviews_for_retrieval_drops_rows_with_empty_required_fields(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    output = tmp_path / "retrieval.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "键盘A",
                "category": "键盘",
                "clean_text": "手感清晰",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "台灯B",
                "category": "台灯",
                "clean_text": "",
            },
        ]
    ).to_csv(source, index=False)

    frame = slim_reviews_for_retrieval(source, output)

    assert frame["review_id"].tolist() == ["R1"]


def test_slim_reviews_for_retrieval_requires_expected_columns(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "键盘A",
                "category": "键盘",
            }
        ]
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Missing required review columns: clean_text"):
        slim_reviews_for_retrieval(source, tmp_path / "retrieval.csv")


def test_slim_reviews_for_retrieval_cli_runs_from_command_line(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    output = tmp_path / "retrieval.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "键盘A",
                "category": "键盘",
                "clean_text": "手感清晰",
                "source": "xiaomi_mall",
            }
        ]
    ).to_csv(source, index=False)

    result = subprocess.run(
        [sys.executable, str(SLIM_SCRIPT), str(source), str(output)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    persisted = pd.read_csv(output, encoding="utf-8-sig")
    assert persisted.columns.tolist() == [
        "review_id",
        "product_id",
        "product_name",
        "category",
        "clean_text",
    ]
