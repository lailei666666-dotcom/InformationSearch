from __future__ import annotations

import sys
import json
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.aggregation import aggregate_review_hits
from src.traditional_retrieval.bm25_engine import BM25Engine


def run_bm25(
    index_path: Path,
    query: str,
    top_k: int = 10,
    *,
    aggregate: bool = False,
    product_top_n: int = 3,
    evidence_top_n: int = 3,
) -> list[dict[str, object]]:
    engine = BM25Engine.load(index_path)
    if not aggregate:
        return engine.search(query, top_k=top_k)

    review_pool_size = max(top_k, product_top_n * max(evidence_top_n, 1) * 3)
    hits = engine.search(query, top_k=review_pool_size)
    return aggregate_review_hits(
        hits,
        product_top_n=product_top_n,
        evidence_top_n=evidence_top_n,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run BM25 retrieval against a serialized review index.")
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--product-top-n", type=int, default=3)
    parser.add_argument("--evidence-top-n", type=int, default=3)
    args = parser.parse_args()

    results = run_bm25(
        args.index_path,
        args.query,
        top_k=args.top_k,
        aggregate=args.aggregate,
        product_top_n=args.product_top_n,
        evidence_top_n=args.evidence_top_n,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
