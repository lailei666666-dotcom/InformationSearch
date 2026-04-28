from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_retrieval.run_hybrid import _load_hits, run_hybrid
from src.hybrid_retrieval.fusion import fuse_ranked_hits, fuse_scores, normalize_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_HYBRID_SCRIPT = PROJECT_ROOT / "scripts" / "run_retrieval" / "run_hybrid.py"


def test_fuse_scores_combines_bm25_and_semantic_scores() -> None:
    fused = fuse_scores(
        bm25_hits={"R1": 0.2, "R2": 0.8},
        semantic_hits={"R1": 0.9, "R2": 0.1},
        alpha=0.75,
    )

    assert fused["R1"] > fused["R2"]


def test_fuse_ranked_hits_merges_review_hits_and_assigns_ranks() -> None:
    bm25_hits = [
        {"review_id": "R1", "product_id": "P1", "score": 0.2, "rank": 2},
        {"review_id": "R2", "product_id": "P2", "score": 0.8, "rank": 1},
    ]
    semantic_hits = [
        {"review_id": "R1", "product_id": "P1", "score": 0.9, "rank": 1},
        {"review_id": "R3", "product_id": "P3", "score": 0.3, "rank": 2},
    ]

    fused = fuse_ranked_hits(bm25_hits, semantic_hits, alpha=0.75, top_k=3)

    assert [item["review_id"] for item in fused] == ["R1", "R2", "R3"]
    assert [item["rank"] for item in fused] == [1, 2, 3]
    assert all("fused_score" in item for item in fused)


def test_normalize_scores_returns_zeroes_for_equal_scores() -> None:
    normalized = normalize_scores({"R1": 7.0, "R2": 7.0})

    assert normalized == {"R1": 0.0, "R2": 0.0}


def test_fuse_ranked_hits_rejects_duplicate_review_ids_within_one_source() -> None:
    bm25_hits = [
        {"review_id": "R1", "product_id": "P1", "score": 0.8, "rank": 1},
        {"review_id": "R1", "product_id": "P1", "score": 0.2, "rank": 2},
    ]

    with pytest.raises(ValueError, match="Duplicate review_id 'R1' found in bm25_hits."):
        fuse_ranked_hits(bm25_hits, [], alpha=0.5, top_k=10)


def test_fuse_ranked_hits_returns_empty_list_for_non_positive_top_k() -> None:
    bm25_hits = [{"review_id": "R1", "product_id": "P1", "score": 0.8, "rank": 1}]
    semantic_hits = [{"review_id": "R1", "product_id": "P1", "score": 0.7, "rank": 1}]

    assert fuse_ranked_hits(bm25_hits, semantic_hits, top_k=0) == []
    assert run_hybrid(bm25_hits, semantic_hits, top_k=-3) == []


def test_load_hits_rejects_non_list_payloads(tmp_path: Path) -> None:
    invalid_hits_path = tmp_path / "invalid.json"
    invalid_hits_path.write_text(json.dumps({"review_id": "R1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a list of hits"):
        _load_hits(invalid_hits_path)


def test_run_hybrid_script_can_run_directly(tmp_path: Path) -> None:
    bm25_hits_path = tmp_path / "bm25.json"
    semantic_hits_path = tmp_path / "semantic.json"
    bm25_hits_path.write_text(
        json.dumps(
            [
                {"review_id": "R1", "product_id": "P1", "score": 0.2, "rank": 2},
                {"review_id": "R2", "product_id": "P2", "score": 0.8, "rank": 1},
            ]
        ),
        encoding="utf-8",
    )
    semantic_hits_path.write_text(
        json.dumps(
            [
                {"review_id": "R1", "product_id": "P1", "score": 0.9, "rank": 1},
                {"review_id": "R3", "product_id": "P3", "score": 0.3, "rank": 2},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(RUN_HYBRID_SCRIPT),
            str(bm25_hits_path),
            str(semantic_hits_path),
            "--alpha",
            "0.75",
            "--top-k",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert [item["review_id"] for item in payload] == ["R1", "R2", "R3"]
