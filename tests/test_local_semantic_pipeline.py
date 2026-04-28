from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.build_index.build_bm25_index import build_bm25_index
from scripts.build_index.build_local_semantic_index import build_local_semantic_index
from scripts.run_retrieval.run_hybrid import run_hybrid_text
from scripts.run_retrieval.run_semantic import run_semantic_text
from src.semantic_retrieval.local_encoder import LocalSemanticEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_LOCAL_SEMANTIC_SCRIPT = PROJECT_ROOT / "scripts" / "build_index" / "build_local_semantic_index.py"
RUN_BM25_SCRIPT = PROJECT_ROOT / "scripts" / "run_retrieval" / "run_bm25.py"


def test_local_semantic_encoder_save_load_round_trip(tmp_path: Path) -> None:
    encoder = LocalSemanticEncoder.fit(
        ["静音键盘适合宿舍", "护眼台灯亮度稳定", "轻薄本适合出差"],
        target_dimension=8,
    )
    output = tmp_path / "encoder.pkl"
    encoder.save(output)

    reloaded = LocalSemanticEncoder.load(output)
    original = encoder.encode_text("静音键盘")
    restored = reloaded.encode_text("静音键盘")

    assert original.shape == restored.shape
    assert (original == restored).all()


def test_build_local_semantic_index_script_runs_from_command_line(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "宿舍打字很安静",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "夜里看书不刺眼",
            },
        ]
    ).to_csv(source, index=False)

    index_output = tmp_path / "semantic-index.json"
    encoder_output = tmp_path / "encoder.pkl"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_LOCAL_SEMANTIC_SCRIPT),
            str(source),
            str(index_output),
            str(encoder_output),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert index_output.exists()
    assert encoder_output.exists()


def test_run_semantic_text_returns_keyboard_hit_first(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
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
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "夜里看书不刺眼",
            },
        ]
    ).to_csv(source, index=False)
    index_output = tmp_path / "semantic-index.json"
    encoder_output = tmp_path / "encoder.pkl"
    build_local_semantic_index(source, index_output, encoder_output, target_dimension=8)

    results = run_semantic_text(
        index_output,
        encoder_output,
        "宿舍安静键盘",
        top_k=2,
    )

    assert results[0]["review_id"] == "R1"


def test_run_hybrid_text_returns_aggregated_keyboard_product_first(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
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
                "clean_text": "键盘回弹清晰 适合夜里写论文",
            },
            {
                "review_id": "R3",
                "product_id": "P2",
                "product_name": "护眼台灯B",
                "category": "台灯",
                "clean_text": "夜里看书不刺眼",
            },
        ]
    ).to_csv(source, index=False)

    bm25_index_output = tmp_path / "bm25-index.json"
    semantic_index_output = tmp_path / "semantic-index.json"
    encoder_output = tmp_path / "encoder.pkl"
    build_bm25_index(source, bm25_index_output)
    build_local_semantic_index(source, semantic_index_output, encoder_output, target_dimension=8)

    results = run_hybrid_text(
        bm25_index_path=bm25_index_output,
        semantic_index_path=semantic_index_output,
        encoder_path=encoder_output,
        query_text="宿舍静音键盘",
        aggregate=True,
        product_top_n=2,
        evidence_top_n=2,
    )

    assert results[0]["product_id"] == "P1"


def test_run_bm25_script_can_run_directly_after_path_fix(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    index_output = tmp_path / "bm25-index.json"
    pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "宿舍打字很安静",
            }
        ]
    ).to_csv(source, index=False)
    build_bm25_index(source, index_output)

    result = subprocess.run(
        [
            sys.executable,
            str(RUN_BM25_SCRIPT),
            str(index_output),
            "静音键盘",
            "--top-k",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload[0]["review_id"] == "R1"
