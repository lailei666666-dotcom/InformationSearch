from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


@dataclass(slots=True)
class LocalSemanticEncoder:
    vectorizer: HashingVectorizer
    tfidf_transformer: TfidfTransformer
    svd: TruncatedSVD | None
    analyzer: str = "char"
    ngram_range: tuple[int, int] = (1, 4)
    max_features: int = 32768

    @classmethod
    def fit(
        cls,
        texts: list[str],
        *,
        analyzer: str = "char",
        ngram_range: tuple[int, int] = (1, 4),
        max_features: int = 32768,
        target_dimension: int = 256,
        random_state: int = 42,
    ) -> "LocalSemanticEncoder":
        if not texts:
            raise ValueError("Cannot fit local semantic encoder on an empty corpus.")

        vectorizer = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            n_features=max_features,
            alternate_sign=False,
            norm=None,
        )
        tfidf_transformer = TfidfTransformer()
        matrix = tfidf_transformer.fit_transform(vectorizer.transform(texts))
        svd = _fit_svd_if_possible(
            matrix_shape=matrix.shape,
            matrix=matrix,
            target_dimension=target_dimension,
            random_state=random_state,
        )
        return cls(
            vectorizer=vectorizer,
            tfidf_transformer=tfidf_transformer,
            svd=svd,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
        )

    @classmethod
    def load(cls, source: Path) -> "LocalSemanticEncoder":
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("kind") != "local_semantic_encoder":
            raise ValueError("Unsupported local semantic encoder artifact.")
        return cls(
            vectorizer=payload["vectorizer"],
            tfidf_transformer=payload["tfidf_transformer"],
            svd=payload["svd"],
            analyzer=payload["analyzer"],
            ngram_range=tuple(payload["ngram_range"]),
            max_features=int(payload["max_features"]),
        )

    def save(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(
                {
                    "kind": "local_semantic_encoder",
                    "vectorizer": self.vectorizer,
                    "tfidf_transformer": self.tfidf_transformer,
                    "svd": self.svd,
                    "analyzer": self.analyzer,
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                },
                handle,
            )

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        matrix = self.tfidf_transformer.transform(self.vectorizer.transform(texts))
        if self.svd is not None:
            dense = self.svd.transform(matrix)
        else:
            dense = matrix.toarray()
        return normalize(np.asarray(dense, dtype="float32"), norm="l2")

    @property
    def dimension(self) -> int:
        if self.svd is not None:
            return int(self.svd.n_components)
        return int(self.max_features)


def _fit_svd_if_possible(
    *,
    matrix_shape: tuple[int, int],
    matrix,
    target_dimension: int,
    random_state: int,
) -> TruncatedSVD | None:
    sample_count, feature_count = matrix_shape
    max_components = min(sample_count - 1, feature_count - 1, target_dimension)
    if max_components < 2:
        return None
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    svd.fit(matrix)
    return svd
