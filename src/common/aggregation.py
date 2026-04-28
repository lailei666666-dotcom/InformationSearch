from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_review_hits(
    hits: list[dict[str, Any]],
    product_top_n: int = 3,
    evidence_top_n: int = 3,
) -> list[dict[str, Any]]:
    if product_top_n <= 0 or evidence_top_n <= 0:
        return []

    grouped_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        product_id = str(hit.get("product_id", "")).strip()
        if not product_id:
            continue
        grouped_hits[product_id].append(dict(hit))

    products: list[dict[str, Any]] = []
    for product_id, product_hits in grouped_hits.items():
        ranked_hits = sorted(
            product_hits,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        evidence = ranked_hits[:evidence_top_n]
        score = sum(float(item.get("score", 0.0)) for item in ranked_hits) / len(ranked_hits)
        top_hit = ranked_hits[0]
        products.append(
            {
                "product_id": product_id,
                "product_name": top_hit.get("product_name"),
                "category": top_hit.get("category"),
                "score": score,
                "evidence": evidence,
            }
        )

    products.sort(key=lambda item: float(item["score"]), reverse=True)
    return products[:product_top_n]
