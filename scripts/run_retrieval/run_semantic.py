from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import numpy as np


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


def run_semantic(
    index_path: Path,
    query_vector: np.ndarray,
    *,
    category: str | None = None,
    top_k: int = 10,
) -> list[dict[str, object]]:
    engine = SemanticEngine.load(index_path)
    return engine.search(query_vector=query_vector, category=category, top_k=top_k)


def run_semantic_text(
    index_path: Path,
    query_text_or_encoder_path: str | Path,
    maybe_query_text: str | None = None,
    *,
    encoder_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    category: str | None = None,
    top_k: int = 10,
    aggregate: bool = False,
    product_top_n: int = 3,
    evidence_top_n: int = 3,
) -> list[dict[str, object]]:
    if maybe_query_text is None:
        query_text = str(query_text_or_encoder_path)
    else:
        if encoder_path is not None:
            raise ValueError("Pass encoder_path either positionally or by keyword, not both.")
        encoder_path = Path(query_text_or_encoder_path)
        query_text = maybe_query_text

    parsed = parse_query(query_text)
    effective_category = category or parsed.category
    retrieval_text = parsed.need_text or query_text.strip()
    query_vector = encode_query_text(
        retrieval_text,
        encoder_path=encoder_path,
        embedding_client=embedding_client,
    )

    review_pool_size = max(top_k, product_top_n * max(evidence_top_n, 1) * 3)
    hits = run_semantic(
        index_path,
        query_vector,
        category=effective_category,
        top_k=review_pool_size if aggregate else top_k,
    )
    if not aggregate:
        return hits
    return aggregate_review_hits(
        hits,
        product_top_n=product_top_n,
        evidence_top_n=evidence_top_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic retrieval against a serialized review index.")
    parser.add_argument("index_path", type=Path)
    parser.add_argument("query_vector", nargs="?", help="JSON array representation of the query embedding vector.")
    parser.add_argument("--query-text", default=None, help="Plain-text query to encode with a saved local encoder.")
    parser.add_argument("--encoder-path", type=Path, default=None)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--category")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--product-top-n", type=int, default=3)
    parser.add_argument("--evidence-top-n", type=int, default=3)
    args = parser.parse_args()

    if args.query_text:
        if args.encoder_path is not None:
            embedding_client = None
        else:
            embedding_client = build_default_embedding_client(args.embedding_cache)
        results = run_semantic_text(
            args.index_path,
            args.query_text,
            encoder_path=args.encoder_path,
            embedding_client=embedding_client,
            category=args.category,
            top_k=args.top_k,
            aggregate=args.aggregate,
            product_top_n=args.product_top_n,
            evidence_top_n=args.evidence_top_n,
        )
    else:
        if args.query_vector is None:
            parser.error("Provide either query_vector or --query-text.")
        query_vector = np.asarray(json.loads(args.query_vector), dtype="float32")
        results = run_semantic(
            args.index_path,
            query_vector,
            category=args.category,
            top_k=args.top_k,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
