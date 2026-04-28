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

from scripts.run_retrieval.run_bm25 import run_bm25  # noqa: E402
from scripts.run_retrieval.run_hybrid import run_hybrid_text  # noqa: E402
from scripts.run_retrieval.run_semantic import run_semantic_text  # noqa: E402
from scripts.visualize.make_tables import build_summary_rows, write_summary_table  # noqa: E402
from src.evaluation.benchmark import evaluate_qrels, summarize_metrics  # noqa: E402
from src.evaluation.qrels import load_product_qrels, load_query_set, load_review_qrels  # noqa: E402
from src.semantic_retrieval.embedding_client import EmbeddingClient  # noqa: E402
from src.semantic_retrieval.runtime import (  # noqa: E402
    DEFAULT_EMBEDDING_CACHE_PATH,
    build_default_embedding_client,
)


SYSTEMS = ("bm25", "semantic", "hybrid")


def run_full_benchmark(
    *,
    annotations_dir: Path,
    bm25_index: Path,
    semantic_index: Path,
    encoder_path: Path | None,
    output_dir: Path,
    embedding_cache: Path | None = None,
    top_k: int = 10,
    product_top_n: int | None = None,
    evidence_top_n: int = 3,
    alpha: float = 0.5,
    min_relevance: int = 1,
) -> dict[str, Path]:
    query_set = load_query_set(annotations_dir)
    review_qrels = load_review_qrels(annotations_dir)
    product_qrels = load_product_qrels(annotations_dir)

    effective_product_top_n = product_top_n or top_k
    ranked_dir = output_dir / "ranked"
    benchmarks_dir = output_dir / "benchmarks"
    tables_dir = output_dir / "tables"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    benchmark_paths: list[Path] = []
    written_paths: dict[str, Path] = {}
    embedding_client = _build_runtime_embedding_client(encoder_path=encoder_path, embedding_cache=embedding_cache)

    for system in SYSTEMS:
        ranked_reviews, ranked_products = _run_system_rankings(
            system=system,
            query_set=query_set,
            bm25_index=bm25_index,
            semantic_index=semantic_index,
            encoder_path=encoder_path,
            embedding_client=embedding_client,
            top_k=top_k,
            product_top_n=effective_product_top_n,
            evidence_top_n=evidence_top_n,
            alpha=alpha,
        )
        review_rank_path = ranked_dir / f"{system}_reviews.json"
        product_rank_path = ranked_dir / f"{system}_products.json"
        _write_json(review_rank_path, ranked_reviews)
        _write_json(product_rank_path, ranked_products)
        written_paths[f"{system}_reviews_ranked"] = review_rank_path
        written_paths[f"{system}_products_ranked"] = product_rank_path

        review_benchmark = _evaluate_ranked_payload(
            ranked_reviews,
            qrels=review_qrels,
            id_column="review_id",
            k=top_k,
            min_relevance=min_relevance,
            target="reviews",
        )
        product_benchmark = _evaluate_ranked_payload(
            ranked_products,
            qrels=product_qrels,
            id_column="product_id",
            k=effective_product_top_n,
            min_relevance=min_relevance,
            target="products",
        )

        review_benchmark_path = benchmarks_dir / f"{system}_reviews.json"
        product_benchmark_path = benchmarks_dir / f"{system}_products.json"
        _write_json(review_benchmark_path, review_benchmark)
        _write_json(product_benchmark_path, product_benchmark)
        benchmark_paths.extend([review_benchmark_path, product_benchmark_path])
        written_paths[f"{system}_reviews_benchmark"] = review_benchmark_path
        written_paths[f"{system}_products_benchmark"] = product_benchmark_path

    summary_path = tables_dir / "benchmark_summary.csv"
    write_summary_table(build_summary_rows(benchmark_paths), summary_path)
    written_paths["summary_table"] = summary_path
    return written_paths


def _run_system_rankings(
    *,
    system: str,
    query_set: Any,
    bm25_index: Path,
    semantic_index: Path,
    encoder_path: Path | None,
    embedding_client: EmbeddingClient | None,
    top_k: int,
    product_top_n: int,
    evidence_top_n: int,
    alpha: float,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ranked_reviews: dict[str, list[str]] = {}
    ranked_products: dict[str, list[str]] = {}

    for row in query_set.itertuples(index=False):
        query_id = str(row.query_id)
        query_text = str(row.query_text)
        category = str(row.category)
        if system == "bm25":
            review_hits = run_bm25(bm25_index, query_text, top_k=top_k)
            product_hits = run_bm25(
                bm25_index,
                query_text,
                top_k=product_top_n,
                aggregate=True,
                product_top_n=product_top_n,
                evidence_top_n=evidence_top_n,
            )
        elif system == "semantic":
            review_hits = run_semantic_text(
                semantic_index,
                query_text,
                encoder_path=encoder_path,
                embedding_client=embedding_client,
                category=category,
                top_k=top_k,
            )
            product_hits = run_semantic_text(
                semantic_index,
                query_text,
                encoder_path=encoder_path,
                embedding_client=embedding_client,
                category=category,
                top_k=product_top_n,
                aggregate=True,
                product_top_n=product_top_n,
                evidence_top_n=evidence_top_n,
            )
        elif system == "hybrid":
            review_hits = run_hybrid_text(
                bm25_index_path=bm25_index,
                semantic_index_path=semantic_index,
                query_text=query_text,
                encoder_path=encoder_path,
                embedding_client=embedding_client,
                category=category,
                alpha=alpha,
                top_k=top_k,
            )
            product_hits = run_hybrid_text(
                bm25_index_path=bm25_index,
                semantic_index_path=semantic_index,
                query_text=query_text,
                encoder_path=encoder_path,
                embedding_client=embedding_client,
                category=category,
                alpha=alpha,
                top_k=product_top_n,
                aggregate=True,
                product_top_n=product_top_n,
                evidence_top_n=evidence_top_n,
            )
        else:
            raise ValueError(f"Unsupported system: {system}")

        ranked_reviews[query_id] = _extract_ranked_ids(review_hits, "review_id", limit=top_k)
        ranked_products[query_id] = _extract_ranked_ids(product_hits, "product_id", limit=product_top_n)

    return ranked_reviews, ranked_products


def _extract_ranked_ids(hits: list[dict[str, Any]], id_field: str, *, limit: int) -> list[str]:
    ranked_ids: list[str] = []
    for hit in hits:
        if id_field not in hit:
            continue
        ranked_ids.append(str(hit[id_field]))
        if len(ranked_ids) >= limit:
            break
    return ranked_ids


def _evaluate_ranked_payload(
    ranked_by_query: dict[str, list[str]],
    *,
    qrels: Any,
    id_column: str,
    k: int,
    min_relevance: int,
    target: str,
) -> dict[str, Any]:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_runtime_embedding_client(
    *,
    encoder_path: Path | None,
    embedding_cache: Path | None,
) -> EmbeddingClient | None:
    if encoder_path is not None:
        return None
    return build_default_embedding_client(embedding_cache or DEFAULT_EMBEDDING_CACHE_PATH)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM25, semantic, and hybrid benchmarks over the annotation set.")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--semantic-index", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--product-top-n", type=int, default=None)
    parser.add_argument("--evidence-top-n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-relevance", type=int, default=1)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    written = run_full_benchmark(
        annotations_dir=args.annotations_dir,
        bm25_index=args.bm25_index,
        semantic_index=args.semantic_index,
        encoder_path=args.encoder_path,
        output_dir=args.output_dir,
        embedding_cache=args.embedding_cache,
        top_k=args.top_k,
        product_top_n=args.product_top_n,
        evidence_top_n=args.evidence_top_n,
        alpha=args.alpha,
        min_relevance=args.min_relevance,
    )
    print(json.dumps({key: str(value) for key, value in written.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
