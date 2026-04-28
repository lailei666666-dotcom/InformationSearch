from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest

from src.evaluation.benchmark import evaluate_qrels, summarize_metrics
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from scripts.evaluate.run_benchmark import _load_ranked_results, run_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate" / "run_benchmark.py"


def test_precision_and_mrr_match_expected_values() -> None:
    ranked = ["R1", "R2", "R3"]
    relevant = {"R2", "R3"}

    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_recall_at_k_uses_relevant_pool_size() -> None:
    ranked = ["R1", "R2", "R3"]
    relevant = {"R2", "R3", "R4"}

    assert recall_at_k(ranked, relevant, 3) == 2 / 3


def test_precision_and_recall_ignore_duplicate_ranked_hits() -> None:
    ranked = ["R1", "R1", "R2"]
    relevant = {"R1", "R2"}

    assert precision_at_k(ranked, relevant, 3) == 2 / 3
    assert recall_at_k(ranked, relevant, 3) == 1.0


def test_top_k_metrics_do_not_overconsume_ranked_iterators() -> None:
    class FailingIterator:
        def __init__(self, values: list[str], fail_at_read: int) -> None:
            self._values = values
            self._fail_at_read = fail_at_read
            self._read_count = 0

        def __iter__(self) -> "FailingIterator":
            return self

        def __next__(self) -> str:
            if self._read_count >= self._fail_at_read:
                raise AssertionError("ranked iterable was over-consumed")
            if self._read_count >= len(self._values):
                raise StopIteration
            value = self._values[self._read_count]
            self._read_count += 1
            return value

    relevant = {"R1", "R2"}

    precision_ranked = FailingIterator(["R1", "R1", "R2"], fail_at_read=2)
    recall_ranked = FailingIterator(["R1", "R1", "R2"], fail_at_read=2)

    assert precision_at_k(precision_ranked, relevant, 2) == 0.5
    assert recall_at_k(recall_ranked, relevant, 2) == 0.5


def test_precision_at_k_treats_missing_ranks_as_non_relevant() -> None:
    ranked = ["R1"]
    relevant = {"R1"}

    assert precision_at_k(ranked, relevant, 3) == 1 / 3


def test_evaluate_qrels_assigns_zero_scores_to_missing_rankings() -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "R2", "relevance": 1, "category": "cat"},
            {"query_id": "Q2", "review_id": "R9", "relevance": 1, "category": "cat"},
        ]
    )

    metrics_frame = evaluate_qrels({"Q1": ["R1", "R2"]}, qrels, id_column="review_id", k=2)

    assert metrics_frame["query_id"].tolist() == ["Q1", "Q2"]
    assert summarize_metrics(metrics_frame) == {
        "precision_at_k": 0.25,
        "recall_at_k": 0.5,
        "mrr": 0.25,
    }


def test_evaluate_qrels_rejects_unknown_ranked_query_ids() -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "R2", "relevance": 1, "category": "cat"},
        ]
    )

    with pytest.raises(ValueError, match="Unknown query_id"):
        evaluate_qrels({"Q1": ["R2"], "Q999": ["R7"]}, qrels, id_column="review_id", k=2)


def test_evaluate_qrels_allows_queries_without_relevant_docs_after_threshold() -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "R2", "relevance": 2, "category": "cat"},
            {"query_id": "Q2", "review_id": "R9", "relevance": 0, "category": "cat"},
        ]
    )

    metrics_frame = evaluate_qrels(
        {"Q1": ["R2"], "Q2": ["R9"]},
        qrels,
        id_column="review_id",
        min_relevance=1,
        k=2,
    )

    assert metrics_frame["query_id"].tolist() == ["Q1", "Q2"]
    assert metrics_frame.to_dict("records") == [
        {
            "query_id": "Q1",
            "precision_at_k": 0.5,
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
        },
        {
            "query_id": "Q2",
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
        },
    ]


def test_evaluate_qrels_normalizes_numeric_ranked_query_ids() -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "101", "review_id": "R2", "relevance": 1, "category": "cat"},
        ]
    )

    metrics_frame = evaluate_qrels({101: ["R1", "R2"]}, qrels, id_column="review_id", k=2)

    assert metrics_frame["query_id"].tolist() == ["101"]
    assert summarize_metrics(metrics_frame) == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mrr": 0.5,
    }


def test_evaluate_qrels_normalizes_numeric_ranked_document_ids() -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "2", "relevance": 1, "category": "cat"},
        ]
    )

    metrics_frame = evaluate_qrels({"Q1": [1, 2]}, qrels, id_column="review_id", k=2)

    assert metrics_frame["query_id"].tolist() == ["Q1"]
    assert summarize_metrics(metrics_frame) == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mrr": 0.5,
    }


@pytest.mark.parametrize(
    ("ranked_by_query", "match"),
    [
        ({"Q1": "R1"}, "list"),
        ({"Q1": ("R1",)}, "list"),
    ],
)
def test_evaluate_qrels_rejects_non_list_ranked_values(
    ranked_by_query: object,
    match: str,
) -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "R1", "relevance": 1, "category": "cat"},
        ]
    )

    with pytest.raises(ValueError, match=match):
        evaluate_qrels(ranked_by_query, qrels, id_column="review_id", k=2)


@pytest.mark.parametrize("doc_id", [None, {"id": "R1"}])
def test_evaluate_qrels_rejects_malformed_ranked_document_ids(doc_id: object) -> None:
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "review_id": "R1", "relevance": 1, "category": "cat"},
        ]
    )

    with pytest.raises(ValueError, match="document ids"):
        evaluate_qrels({"Q1": [doc_id]}, qrels, id_column="review_id", k=2)


def test_run_benchmark_script_help_works_from_absolute_path(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "ranked_path" in result.stdout


def test_run_benchmark_rejects_invalid_target(tmp_path: Path) -> None:
    ranked_path = tmp_path / "ranked.json"
    ranked_path.write_text(json.dumps({"Q1": ["R1"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported target"):
        run_benchmark(ranked_path, target="invalid")


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (["R1", "R2"], "mapping"),
        ({"Q1": "R1"}, "list"),
    ],
)
def test_load_ranked_results_rejects_malformed_json(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    ranked_path = tmp_path / "ranked.json"
    ranked_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_ranked_results(ranked_path)


@pytest.mark.parametrize("doc_id", [None, {"id": "R1"}])
def test_load_ranked_results_rejects_malformed_document_ids(
    tmp_path: Path,
    doc_id: object,
) -> None:
    ranked_path = tmp_path / "ranked.json"
    ranked_path.write_text(json.dumps({"Q1": [doc_id]}), encoding="utf-8")

    with pytest.raises(ValueError, match="document ids"):
        _load_ranked_results(ranked_path)
