from __future__ import annotations

import sys
import argparse
import json
from numbers import Real
from pathlib import Path

def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.evaluation.benchmark import evaluate_qrels, summarize_metrics
from src.evaluation.qrels import load_product_qrels, load_review_qrels


def run_benchmark(
    ranked_path: Path,
    *,
    target: str = "reviews",
    k: int = 10,
    min_relevance: int = 1,
) -> dict[str, object]:
    if target not in {"reviews", "products"}:
        raise ValueError(f"Unsupported target: {target}")

    ranked_by_query = _load_ranked_results(ranked_path)
    qrels = load_review_qrels() if target == "reviews" else load_product_qrels()
    id_column = "review_id" if target == "reviews" else "product_id"
    metrics_frame = evaluate_qrels(
        ranked_by_query,
        qrels,
        id_column=id_column,
        min_relevance=min_relevance,
        k=k,
    )
    return {
        "target": target,
        "k": k,
        "min_relevance": min_relevance,
        "summary": summarize_metrics(metrics_frame),
        "per_query": metrics_frame.to_dict(orient="records"),
    }


def _load_ranked_results(ranked_path: Path) -> dict[str, list[str]]:
    payload = json.loads(ranked_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ranked results JSON must be an object mapping query_id to a list of ids.")

    normalized: dict[str, list[str]] = {}
    for query_id, ranked_ids in payload.items():
        if not isinstance(ranked_ids, list):
            raise ValueError(
                "Ranked results JSON must map each query_id to a list of ids."
            )
        normalized[str(query_id)] = [_normalize_ranked_doc_id(doc_id) for doc_id in ranked_ids]
    return normalized


def _normalize_ranked_doc_id(doc_id: object) -> str:
    if isinstance(doc_id, str):
        return doc_id
    if isinstance(doc_id, Real) and not isinstance(doc_id, bool):
        return str(doc_id)
    raise ValueError("Ranked results JSON document ids must be strings or numeric scalars.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked retrieval outputs against qrels.")
    parser.add_argument("ranked_path", type=Path, help="JSON mapping query_id to ranked document IDs.")
    parser.add_argument("--target", choices=("reviews", "products"), default="reviews")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-relevance", type=int, default=1)
    args = parser.parse_args()

    result = run_benchmark(
        args.ranked_path,
        target=args.target,
        k=args.k,
        min_relevance=args.min_relevance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
