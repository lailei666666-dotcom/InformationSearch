from __future__ import annotations

from typing import Any


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    values = [float(score) for score in scores.values()]
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {review_id: 0.0 for review_id in scores}

    scale = maximum - minimum
    return {
        review_id: (float(score) - minimum) / scale
        for review_id, score in scores.items()
    }


def fuse_scores(
    *,
    bm25_hits: dict[str, float],
    semantic_hits: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    normalized_bm25 = normalize_scores(bm25_hits)
    normalized_semantic = normalize_scores(semantic_hits)
    review_ids = set(normalized_bm25) | set(normalized_semantic)

    return {
        review_id: alpha * normalized_semantic.get(review_id, 0.0)
        + (1.0 - alpha) * normalized_bm25.get(review_id, 0.0)
        for review_id in review_ids
    }


def fuse_ranked_hits(
    bm25_hits: list[dict[str, Any]],
    semantic_hits: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
    top_k: int | None = 10,
) -> list[dict[str, Any]]:
    if top_k is not None and top_k <= 0:
        return []

    bm25_by_id = _index_hits_by_review_id(bm25_hits, source_name="bm25_hits")
    semantic_by_id = _index_hits_by_review_id(semantic_hits, source_name="semantic_hits")
    fused_scores = fuse_scores(
        bm25_hits={review_id: float(hit["score"]) for review_id, hit in bm25_by_id.items()},
        semantic_hits={review_id: float(hit["score"]) for review_id, hit in semantic_by_id.items()},
        alpha=alpha,
    )

    ranked_review_ids = sorted(
        fused_scores,
        key=lambda review_id: (-fused_scores[review_id], review_id),
    )
    if top_k is not None:
        ranked_review_ids = ranked_review_ids[:top_k]

    fused_hits: list[dict[str, Any]] = []
    for rank, review_id in enumerate(ranked_review_ids, start=1):
        base_hit = dict(bm25_by_id.get(review_id) or semantic_by_id[review_id])
        base_hit["bm25_score"] = float(bm25_by_id.get(review_id, {}).get("score", 0.0))
        base_hit["semantic_score"] = float(semantic_by_id.get(review_id, {}).get("score", 0.0))
        base_hit["fused_score"] = fused_scores[review_id]
        base_hit["score"] = fused_scores[review_id]
        base_hit["rank"] = rank
        fused_hits.append(base_hit)
    return fused_hits


def _index_hits_by_review_id(
    hits: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for hit in hits:
        review_id = str(hit["review_id"])
        if review_id in indexed:
            raise ValueError(f"Duplicate review_id '{review_id}' found in {source_name}.")
        indexed[review_id] = hit
    return indexed
