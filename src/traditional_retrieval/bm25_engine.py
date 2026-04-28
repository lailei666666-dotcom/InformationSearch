from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from src.traditional_retrieval.tokenizer import tokenize


REQUIRED_COLUMNS = ("review_id", "product_id", "product_name", "category", "clean_text")


@dataclass(slots=True)
class BM25Engine:
    documents: list[dict[str, Any]]
    corpus_tokens: list[list[str]]
    indexed_texts: list[str]
    _bm25: BM25Okapi = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bm25 = BM25Okapi(self.corpus_tokens)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BM25Engine":
        _require_columns(frame, REQUIRED_COLUMNS)
        _require_non_empty_values(frame, REQUIRED_COLUMNS)

        records = frame.loc[:, list(REQUIRED_COLUMNS)].to_dict(orient="records")
        indexed_texts = [_build_indexed_text(record) for record in records]
        corpus_tokens = [_tokenize_document(text) for text in indexed_texts]
        return cls(documents=records, corpus_tokens=corpus_tokens, indexed_texts=indexed_texts)

    @classmethod
    def load(cls, source: Path) -> "BM25Engine":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            documents=payload["documents"],
            corpus_tokens=payload["corpus_tokens"],
            indexed_texts=payload["indexed_texts"],
        )

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "documents": self.documents,
                    "corpus_tokens": self.corpus_tokens,
                    "indexed_texts": self.indexed_texts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        query_terms = set(query_tokens)
        for index in ranked_indices:
            if not query_terms.intersection(self.corpus_tokens[index]):
                continue
            hit = dict(self.documents[index])
            hit["score"] = float(scores[index])
            hit["rank"] = len(results) + 1
            results.append(hit)
            if len(results) >= top_k:
                break
        return results


def _build_indexed_text(record: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(record.get("product_name", "")),
            str(record.get("category", "")),
            str(record.get("clean_text", "")),
        )
        if part
    )


def _tokenize_document(text: str) -> list[str]:
    tokens = tokenize(text)
    return tokens or ["__empty__"]


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def _require_non_empty_values(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    invalid_columns: list[str] = []
    for column in required_columns:
        values = frame[column]
        if values.isna().any():
            invalid_columns.append(column)
            continue
        if values.map(lambda value: not str(value).strip()).any():
            invalid_columns.append(column)

    if invalid_columns:
        invalid_display = ", ".join(sorted(invalid_columns))
        raise ValueError(f"Required review columns contain empty values: {invalid_display}")
