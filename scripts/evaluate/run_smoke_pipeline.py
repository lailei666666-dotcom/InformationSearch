from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Any

def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.aggregation import aggregate_review_hits
from src.common.io_utils import read_table
from src.evaluation.qrels import load_query_set
from src.traditional_retrieval.bm25_engine import BM25Engine


def run_smoke_pipeline(
    processed_path: Path,
    output_dir: Path,
    *,
    query: str | None = None,
    top_k: int = 10,
    product_top_n: int = 3,
    evidence_top_n: int = 2,
) -> dict[str, Any]:
    frame = read_table(processed_path)
    engine = BM25Engine.from_frame(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "indexes" / "bm25_smoke_index.json"
    results_path = output_dir / "runs" / "bm25_smoke_results.json"
    summary_path = output_dir / "smoke_summary.json"

    index_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save(index_path)

    resolved_query = _resolve_query(query)
    review_hits = engine.search(resolved_query, top_k=max(top_k, product_top_n * evidence_top_n * 3))
    product_hits = aggregate_review_hits(
        review_hits,
        product_top_n=product_top_n,
        evidence_top_n=evidence_top_n,
    )
    results_path.write_text(json.dumps(product_hits, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "processed_path": str(processed_path),
        "row_count": int(len(frame)),
        "query": resolved_query,
        "review_hit_count": len(review_hits),
        "result_count": len(product_hits),
        "index_path": str(index_path),
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _resolve_query(query: str | None) -> str:
    if query and query.strip():
        return query.strip()

    query_set = load_query_set()
    if query_set.empty:
        raise ValueError("Query set is empty; provide --query explicitly.")

    query_text = str(query_set.iloc[0]["query_text"]).strip()
    if not query_text:
        raise ValueError("First query in query set is empty; provide --query explicitly.")
    return query_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small BM25 smoke pipeline on a processed review dataset.")
    parser.add_argument("processed", type=Path, help="Processed review CSV path.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for smoke artifacts.")
    parser.add_argument("--query", default=None, help="Optional smoke query; defaults to the first annotated query.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--product-top-n", type=int, default=3)
    parser.add_argument("--evidence-top-n", type=int, default=2)
    args = parser.parse_args()

    summary = run_smoke_pipeline(
        args.processed,
        args.output,
        query=args.query,
        top_k=args.top_k,
        product_top_n=args.product_top_n,
        evidence_top_n=args.evidence_top_n,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
