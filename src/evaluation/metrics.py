from __future__ import annotations

from collections.abc import Iterable
from itertools import islice


def precision_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    ranked_at_k = list(islice(ranked, k))
    if not ranked_at_k:
        return 0.0
    hits = _count_unique_hits(ranked_at_k, relevant)
    return hits / k


def recall_at_k(ranked: Iterable[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    ranked_at_k = list(islice(ranked, k))
    hits = _count_unique_hits(ranked_at_k, relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: Iterable[str], relevant: set[str]) -> float:
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def _count_unique_hits(ranked: Iterable[str], relevant: set[str]) -> int:
    seen: set[str] = set()
    hits = 0
    for doc_id in ranked:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in relevant:
            hits += 1
    return hits
