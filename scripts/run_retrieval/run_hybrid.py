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

from src.common.aggregation import aggregate_review_hits  # noqa: E402
from src.common.query_parser import parse_query  # noqa: E402
from src.semantic_retrieval.embedding_client import EmbeddingClient  # noqa: E402
from src.semantic_retrieval.runtime import (  # noqa: E402
    DEFAULT_EMBEDDING_CACHE_PATH,
    build_default_embedding_client,
    encode_query_text,
)
from src.semantic_retrieval.semantic_engine import SemanticEngine  # noqa: E402
from src.traditional_retrieval.bm25_engine import BM25Engine  # noqa: E402
from src.hybrid_retrieval.fusion import fuse_ranked_hits  # noqa: E402


def run_hybrid(
    bm25_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    return fuse_ranked_hits(
        bm25_hits,
        semantic_hits,
        alpha=alpha,
        top_k=top_k,
    )


def run_hybrid_text(
    *,
    bm25_index_path: Path,
    semantic_index_path: Path,
    query_text: str,
    encoder_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    category: str | None = None,
    alpha: float = 0.5,
    top_k: int = 10,
    aggregate: bool = False,
    product_top_n: int = 3,
    evidence_top_n: int = 3,
) -> list[dict[str, Any]]:
    parsed = parse_query(query_text)
    effective_category = category or parsed.category
    retrieval_text = parsed.need_text or query_text.strip()

    bm25_engine = BM25Engine.load(bm25_index_path)
    review_pool_size = max(top_k, product_top_n * max(evidence_top_n, 1) * 3)
    bm25_hits = bm25_engine.search(query_text, top_k=review_pool_size if aggregate else top_k)
    bm25_hits = _filter_hits_by_category(bm25_hits, effective_category)

    semantic_engine = SemanticEngine.load(semantic_index_path)
    query_vector = encode_query_text(
        retrieval_text,
        encoder_path=encoder_path,
        embedding_client=embedding_client,
    )
    semantic_hits = semantic_engine.search(
        query_vector=query_vector,
        category=effective_category,
        top_k=review_pool_size if aggregate else top_k,
    )

    fused_review_hits = run_hybrid(
        bm25_hits,
        semantic_hits,
        alpha=alpha,
        top_k=review_pool_size if aggregate else top_k,
    )
    if not aggregate:
        return fused_review_hits
    return aggregate_review_hits(
        fused_review_hits,
        product_top_n=product_top_n,
        evidence_top_n=evidence_top_n,
    )


def _filter_hits_by_category(
    hits: list[dict[str, Any]],
    category: str | None,
) -> list[dict[str, Any]]:
    if category is None:
        return hits
    normalized = category.strip()
    if not normalized:
        return hits
    return [hit for hit in hits if str(hit.get("category", "")).strip() == normalized]


def _load_hits(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of hits in {path}.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse BM25 and semantic review hits.")
    parser.add_argument("bm25_hits_path", nargs="?", type=Path)
    parser.add_argument("semantic_hits_path", nargs="?", type=Path)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--bm25-index", type=Path, default=None)
    parser.add_argument("--semantic-index", type=Path, default=None)
    parser.add_argument("--encoder-path", type=Path, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--category", default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--product-top-n", type=int, default=3)
    parser.add_argument("--evidence-top-n", type=int, default=3)
    args = parser.parse_args()

    if args.query_text:
        if args.bm25_index is None or args.semantic_index is None:
            parser.error("--bm25-index and --semantic-index are required with --query-text.")
        if args.encoder_path is not None:
            embedding_client = None
        else:
            embedding_client = build_default_embedding_client(args.embedding_cache)
        results = run_hybrid_text(
            bm25_index_path=args.bm25_index,
            semantic_index_path=args.semantic_index,
            query_text=args.query_text,
            encoder_path=args.encoder_path,
            embedding_client=embedding_client,
            category=args.category,
            alpha=args.alpha,
            top_k=args.top_k,
            aggregate=args.aggregate,
            product_top_n=args.product_top_n,
            evidence_top_n=args.evidence_top_n,
        )
    else:
        if args.bm25_hits_path is None or args.semantic_hits_path is None:
            parser.error("Provide bm25_hits_path and semantic_hits_path, or use --query-text mode.")
        results = run_hybrid(
            _load_hits(args.bm25_hits_path),
            _load_hits(args.semantic_hits_path),
            alpha=args.alpha,
            top_k=args.top_k,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
