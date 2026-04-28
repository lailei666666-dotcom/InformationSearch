from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.build_index.build_bm25_index import build_bm25_index
from scripts.build_index.build_local_semantic_index import build_local_semantic_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_FULL_BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate" / "run_full_benchmark.py"


def test_run_full_benchmark_script_creates_rankings_and_benchmarks(tmp_path: Path) -> None:
    processed = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "宿舍打字很安静 手感舒服",
            },
            {
                "review_id": "R2",
                "product_id": "P1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "键盘打字顺手 办公声音不大",
            },
            {
                "review_id": "R3",
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "护眼台灯 看书久了眼睛不累",
            },
            {
                "review_id": "R4",
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "桌面不占地方 亮度足够",
            },
        ]
    ).to_csv(processed, index=False, encoding="utf-8-sig")

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"query_id": "Q001", "query_type": "关键词明确型", "category": "键盘", "query_text": "宿舍安静键盘"},
            {"query_id": "Q002", "query_type": "关键词明确型", "category": "台灯", "query_text": "护眼台灯"},
        ]
    ).to_csv(annotations_dir / "queries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"query_id": "Q001", "review_id": "R1", "relevance": 2, "category": "键盘"},
            {"query_id": "Q001", "review_id": "R2", "relevance": 2, "category": "键盘"},
            {"query_id": "Q002", "review_id": "R3", "relevance": 2, "category": "台灯"},
            {"query_id": "Q002", "review_id": "R4", "relevance": 2, "category": "台灯"},
        ]
    ).to_csv(annotations_dir / "qrels_reviews.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"query_id": "Q001", "product_id": "P1", "relevance": 3, "category": "键盘"},
            {"query_id": "Q002", "product_id": "P2", "relevance": 3, "category": "台灯"},
        ]
    ).to_csv(annotations_dir / "qrels_products.csv", index=False, encoding="utf-8-sig")

    bm25_index = tmp_path / "bm25.json"
    semantic_index = tmp_path / "semantic.json"
    encoder_path = tmp_path / "encoder.pkl"
    build_bm25_index(processed, bm25_index)
    build_local_semantic_index(processed, semantic_index, encoder_path, target_dimension=8)

    output_dir = tmp_path / "benchmark-output"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_FULL_BENCHMARK_SCRIPT),
            "--annotations-dir",
            str(annotations_dir),
            "--bm25-index",
            str(bm25_index),
            "--semantic-index",
            str(semantic_index),
            "--encoder-path",
            str(encoder_path),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0

    summary_path = output_dir / "tables" / "benchmark_summary.csv"
    bm25_review_benchmark = output_dir / "benchmarks" / "bm25_reviews.json"
    hybrid_product_ranking = output_dir / "ranked" / "hybrid_products.json"

    assert summary_path.exists()
    assert bm25_review_benchmark.exists()
    assert hybrid_product_ranking.exists()

    summary = pd.read_csv(summary_path)
    bm25_payload = json.loads(bm25_review_benchmark.read_text(encoding="utf-8"))
    hybrid_ranked = json.loads(hybrid_product_ranking.read_text(encoding="utf-8"))

    assert set(summary["system"]) >= {"bm25_reviews", "semantic_reviews", "hybrid_reviews"}
    assert bm25_payload["summary"]["mrr"] >= 0.0
    assert hybrid_ranked["Q001"][0] == "P1"
