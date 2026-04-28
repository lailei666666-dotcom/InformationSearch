from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from scripts.visualize.make_figures import write_metric_bar_chart
from scripts.visualize.make_tables import build_summary_rows, write_summary_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_SCRIPT = PROJECT_ROOT / "scripts" / "visualize" / "make_tables.py"
FIGURES_SCRIPT = PROJECT_ROOT / "scripts" / "visualize" / "make_figures.py"
TEST_TMP_ROOT = PROJECT_ROOT / ".tmp" / "report-tests"


def test_write_summary_table_creates_csv() -> None:
    temp_dir = _make_temp_dir()
    output = temp_dir / "tables" / "summary.csv"
    write_summary_table(
        rows=[{"system": "bm25", "precision_at_k": 0.4, "mrr": 0.5}],
        output=output,
    )

    assert output.exists()
    frame = pd.read_csv(output)
    assert frame.to_dict(orient="records") == [{"system": "bm25", "precision_at_k": 0.4, "mrr": 0.5}]


def test_build_summary_rows_flattens_benchmark_payloads() -> None:
    temp_dir = _make_temp_dir()
    benchmark_path = temp_dir / "bm25_reviews.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "target": "reviews",
                "k": 10,
                "min_relevance": 1,
                "summary": {"precision_at_k": 0.4, "recall_at_k": 0.6, "mrr": 0.5},
                "per_query": [],
            }
        ),
        encoding="utf-8",
    )

    rows = build_summary_rows([benchmark_path], labels={"bm25_reviews": "bm25"})

    assert rows == [
        {
            "system": "bm25",
            "target": "reviews",
            "k": 10,
            "min_relevance": 1,
            "precision_at_k": 0.4,
            "recall_at_k": 0.6,
            "mrr": 0.5,
            "source_file": str(benchmark_path),
        }
    ]


def test_build_summary_rows_accepts_utf8_bom_json() -> None:
    temp_dir = _make_temp_dir()
    benchmark_path = temp_dir / "semantic_reviews.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "target": "reviews",
                "k": 10,
                "min_relevance": 1,
                "summary": {"precision_at_k": 0.5, "recall_at_k": 0.7, "mrr": 0.6},
                "per_query": [],
            }
        ),
        encoding="utf-8-sig",
    )

    rows = build_summary_rows([benchmark_path])

    assert rows[0]["system"] == "semantic_reviews"
    assert rows[0]["mrr"] == 0.6


def test_write_metric_bar_chart_creates_png() -> None:
    temp_dir = _make_temp_dir()
    summary_path = temp_dir / "summary.csv"
    pd.DataFrame(
        [
            {"system": "bm25", "mrr": 0.42},
            {"system": "semantic", "mrr": 0.61},
        ]
    ).to_csv(summary_path, index=False)

    output = temp_dir / "figures" / "mrr.png"
    write_metric_bar_chart(summary_path, metric="mrr", output=output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_visualization_scripts_show_help_from_absolute_path() -> None:
    temp_dir = _make_temp_dir()
    for script in (TABLES_SCRIPT, FIGURES_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            cwd=temp_dir,
        )

        assert result.returncode == 0
        assert "--output" in result.stdout


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
