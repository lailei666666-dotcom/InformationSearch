from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real

import pandas as pd

from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def evaluate_ranked_list(
    ranked_ids: Iterable[str],
    relevant_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    ranked = list(ranked_ids)
    return {
        "precision_at_k": precision_at_k(ranked, relevant_ids, k),
        "recall_at_k": recall_at_k(ranked, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(ranked, relevant_ids),
    }


def evaluate_qrels(
    ranked_by_query: Mapping[str, Iterable[str]],
    qrels: pd.DataFrame,
    *,
    id_column: str,
    min_relevance: int = 1,
    k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    normalized_ranked_by_query = _normalize_ranked_by_query(ranked_by_query)
    query_ids = sorted(qrels["query_id"].astype(str).unique())
    relevant_by_query = _group_relevant_ids(qrels, id_column=id_column, min_relevance=min_relevance)
    unknown_query_ids = sorted(set(normalized_ranked_by_query) - set(query_ids))
    if unknown_query_ids:
        formatted_ids = ", ".join(unknown_query_ids)
        raise ValueError(f"Unknown query_id(s) in ranked results: {formatted_ids}")

    for query_id in query_ids:
        ranked_ids = normalized_ranked_by_query.get(query_id, [])
        metrics = evaluate_ranked_list(ranked_ids, relevant_by_query.get(query_id, set()), k=k)
        rows.append({"query_id": query_id, **metrics})

    return pd.DataFrame(rows)


def summarize_metrics(metrics_frame: pd.DataFrame) -> dict[str, float]:
    if metrics_frame.empty:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "precision_at_k": float(metrics_frame["precision_at_k"].mean()),
        "recall_at_k": float(metrics_frame["recall_at_k"].mean()),
        "mrr": float(metrics_frame["reciprocal_rank"].mean()),
    }


def _group_relevant_ids(
    qrels: pd.DataFrame,
    *,
    id_column: str,
    min_relevance: int,
) -> dict[str, set[str]]:
    filtered = qrels.loc[qrels["relevance"] >= min_relevance, ["query_id", id_column]]
    grouped = filtered.groupby("query_id")[id_column]
    return {str(query_id): set(ids.astype(str)) for query_id, ids in grouped}


def _normalize_ranked_by_query(
    ranked_by_query: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for query_id, ranked_ids in ranked_by_query.items():
        if not isinstance(ranked_ids, list):
            raise ValueError("Ranked results must map each query_id to a list of ids.")
        normalized[str(query_id)] = [_normalize_ranked_doc_id(doc_id) for doc_id in ranked_ids]
    return normalized


def _normalize_ranked_doc_id(doc_id: object) -> str:
    if isinstance(doc_id, str):
        return doc_id
    if isinstance(doc_id, Real) and not isinstance(doc_id, bool):
        return str(doc_id)
    raise ValueError("Ranked results document ids must be strings or numeric scalars.")
