from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from scripts.evaluate.run_smoke_pipeline import run_smoke_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate" / "run_smoke_pipeline.py"
TEST_TMP_ROOT = PROJECT_ROOT / ".tmp" / "smoke-tests"


def test_smoke_pipeline_has_help_output() -> None:
    temp_dir = _make_temp_dir()
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=temp_dir,
    )

    assert result.returncode == 0
    assert "processed" in result.stdout
    assert "--output" in result.stdout


def test_run_smoke_pipeline_creates_summary_and_results() -> None:
    temp_dir = _make_temp_dir()
    processed_path = temp_dir / "processed_reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "机械键盘A",
                "category": "键盘",
                "clean_text": "机械键盘 手感很好 打字舒服",
            },
            {
                "review_id": "R2",
                "product_id": "P1",
                "product_name": "机械键盘A",
                "category": "键盘",
                "clean_text": "键盘 声音清脆 适合办公",
            },
            {
                "review_id": "R3",
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "台灯 亮度稳定 不刺眼",
            },
        ]
    ).to_csv(processed_path, index=False, encoding="utf-8-sig")

    output_dir = temp_dir / "smoke-output"
    summary = run_smoke_pipeline(processed_path, output_dir, query="机械键盘")

    summary_path = output_dir / "smoke_summary.json"
    results_path = output_dir / "runs" / "bm25_smoke_results.json"
    index_path = output_dir / "indexes" / "bm25_smoke_index.json"

    assert summary_path.exists()
    assert results_path.exists()
    assert index_path.exists()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_results = json.loads(results_path.read_text(encoding="utf-8"))

    assert summary["query"] == "机械键盘"
    assert saved_summary["row_count"] == 3
    assert saved_summary["result_count"] >= 1
    assert saved_results[0]["product_id"] == "P1"


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
