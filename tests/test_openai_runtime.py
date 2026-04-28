from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_index.build_bm25_index import build_bm25_index
from scripts.build_index.build_faiss_index import build_faiss_index
from scripts.run_retrieval.run_hybrid import run_hybrid_text
from scripts.run_retrieval.run_semantic import run_semantic_text
from src.semantic_retrieval.embedding_cache import EmbeddingCache
from src.semantic_retrieval.embedding_client import EmbeddingClient
from src.semantic_retrieval.runtime import cache_namespace
from src.semantic_retrieval.semantic_engine import SemanticEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_FULL_BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate" / "run_full_benchmark.py"


def test_run_semantic_text_supports_embedding_client_query_encoding(tmp_path: Path) -> None:
    source = _write_reviews_csv(
        tmp_path / "reviews.csv",
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Quiet Keyboard",
                "category": "键盘",
                "clean_text": "quiet keyboard for dorm typing",
            },
            {
                "review_id": "R2",
                "product_id": "P2",
                "product_name": "Desk Lamp",
                "category": "台灯",
                "clean_text": "soft desk lamp for night reading",
            },
        ],
    )
    _build_semantic_index(
        source,
        tmp_path / "semantic-index.json",
        vectors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"),
    )
    client = EmbeddingClient(embedder=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    results = run_semantic_text(
        tmp_path / "semantic-index.json",
        "quiet dorm keyboard",
        embedding_client=client,
        category="键盘",
        top_k=2,
    )

    assert results[0]["review_id"] == "R1"


def test_run_hybrid_text_supports_embedding_client_for_aggregation(tmp_path: Path) -> None:
    source = _write_reviews_csv(
        tmp_path / "reviews.csv",
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Quiet Keyboard",
                "category": "键盘",
                "clean_text": "quiet keyboard for dorm typing",
            },
            {
                "review_id": "R2",
                "product_id": "P1",
                "product_name": "Quiet Keyboard",
                "category": "键盘",
                "clean_text": "low noise and comfortable keys",
            },
            {
                "review_id": "R3",
                "product_id": "P2",
                "product_name": "Desk Lamp",
                "category": "台灯",
                "clean_text": "compact lamp for bedside reading",
            },
        ],
    )
    bm25_index = tmp_path / "bm25-index.json"
    semantic_index = tmp_path / "semantic-index.json"
    build_bm25_index(source, bm25_index)
    _build_semantic_index(
        source,
        semantic_index,
        vectors=np.asarray([[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0]], dtype="float32"),
    )
    client = EmbeddingClient(embedder=lambda texts: [[1.0, 0.0, 0.0] for _ in texts])

    results = run_hybrid_text(
        bm25_index_path=bm25_index,
        semantic_index_path=semantic_index,
        query_text="need a quiet dorm keyboard",
        embedding_client=client,
        category="键盘",
        aggregate=True,
        product_top_n=2,
        evidence_top_n=2,
    )

    assert results[0]["product_id"] == "P1"


def test_run_full_benchmark_script_supports_openai_cache_mode(tmp_path: Path) -> None:
    source = _write_reviews_csv(
        tmp_path / "reviews.csv",
        [
            {
                "review_id": "R1",
                "product_id": "P1",
                "product_name": "Quiet Keyboard",
                "category": "键盘",
                "clean_text": "quiet keyboard for dorm typing",
            },
            {
                "review_id": "R2",
                "product_id": "P1",
                "product_name": "Quiet Keyboard",
                "category": "键盘",
                "clean_text": "low noise and comfortable keys",
            },
            {
                "review_id": "R3",
                "product_id": "P2",
                "product_name": "Desk Lamp",
                "category": "台灯",
                "clean_text": "soft light for night reading",
            },
            {
                "review_id": "R4",
                "product_id": "P2",
                "product_name": "Desk Lamp",
                "category": "台灯",
                "clean_text": "compact lamp for bedside reading",
            },
        ],
    )

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"query_id": "Q001", "query_type": "需求型", "category": "键盘", "query_text": "quiet dorm keyboard"},
            {"query_id": "Q002", "query_type": "需求型", "category": "台灯", "query_text": "soft bedside lamp"},
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

    bm25_index = tmp_path / "bm25-index.json"
    embeddings_jsonl = tmp_path / "embeddings.jsonl"
    semantic_index = tmp_path / "semantic-index.json"
    cache_path = tmp_path / "embedding-cache.jsonl"
    output_dir = tmp_path / "benchmark-output"
    build_bm25_index(source, bm25_index)
    _write_embedding_rows(
        embeddings_jsonl,
        [
            ("R1", [1.0] + [0.0] * 1535),
            ("R2", [0.98, 0.02] + [0.0] * 1534),
            ("R3", [0.0, 1.0] + [0.0] * 1534),
            ("R4", [0.0, 0.98, 0.02] + [0.0] * 1533),
        ],
    )
    build_faiss_index(source, embeddings_jsonl, semantic_index)

    namespace = cache_namespace("openai_compatible", "text-embedding-v4", 1536)
    cache = EmbeddingCache(cache_path, namespace=namespace)
    cache.set("quiet dorm keyboard", [1.0] + [0.0] * 1535)
    cache.set("soft bedside lamp", [0.0, 1.0] + [0.0] * 1534)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "test-key"
    env["PYTHONUTF8"] = "1"
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
            "--embedding-cache",
            str(cache_path),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert result.returncode == 0
    assert (output_dir / "tables" / "benchmark_summary.csv").exists()
    assert (output_dir / "benchmarks" / "semantic_reviews.json").exists()
    assert (output_dir / "ranked" / "hybrid_products.json").exists()


def _write_reviews_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _build_semantic_index(source: Path, output: Path, *, vectors: np.ndarray) -> None:
    frame = pd.read_csv(source)
    engine = SemanticEngine.from_vectors(frame, vectors)
    engine.save(output)


def _write_embedding_rows(path: Path, rows: list[tuple[str, list[float]]]) -> None:
    payload = [
        {"review_id": review_id, "vector": vector}
        for review_id, vector in rows
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
