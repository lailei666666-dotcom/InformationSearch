from __future__ import annotations

import sys
import json
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.io_utils import read_table  # noqa: E402
from src.semantic_retrieval.local_encoder import LocalSemanticEncoder  # noqa: E402
from src.semantic_retrieval.semantic_engine import SemanticEngine  # noqa: E402

REQUIRED_COLUMNS = ("review_id", "product_id", "product_name", "category", "clean_text")


def build_local_semantic_index(
    source: Path,
    index_output: Path,
    encoder_output: Path,
    *,
    embeddings_output: Path | None = None,
    target_dimension: int = 256,
) -> SemanticEngine:
    frame = read_table(source)
    _require_columns(frame, REQUIRED_COLUMNS)
    documents = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    documents["clean_text"] = documents["clean_text"].map(lambda value: str(value).strip())
    documents = documents.loc[documents["clean_text"].map(bool)].reset_index(drop=True)

    encoder = LocalSemanticEncoder.fit(
        documents["clean_text"].tolist(),
        target_dimension=target_dimension,
    )
    vectors = encoder.encode_texts(documents["clean_text"].tolist())
    metadata = {
        "source_row_count": int(len(frame)),
        "indexed_row_count": int(len(documents)),
        "vector_count": int(len(vectors)),
        "review_ids": documents["review_id"].astype(str).tolist(),
        "source_text_column": "clean_text",
        "embedding_source_format": "jsonl",
        "encoder_type": "local_semantic_encoder",
    }
    engine = SemanticEngine.from_vectors(documents, vectors, metadata=metadata)
    engine.save(index_output)
    encoder.save(encoder_output)

    if embeddings_output is not None:
        embeddings_output.parent.mkdir(parents=True, exist_ok=True)
        with embeddings_output.open("w", encoding="utf-8") as handle:
            for review_id, text, vector in zip(
                documents["review_id"].astype(str).tolist(),
                documents["clean_text"].tolist(),
                vectors.tolist(),
                strict=True,
            ):
                handle.write(
                    json.dumps(
                        {
                            "review_id": review_id,
                            "text": text,
                            "vector": vector,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return engine


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a local semantic index and matching query encoder.")
    parser.add_argument("source", type=Path)
    parser.add_argument("index_output", type=Path)
    parser.add_argument("encoder_output", type=Path)
    parser.add_argument("--embeddings-output", type=Path, default=None)
    parser.add_argument("--target-dimension", type=int, default=256)
    args = parser.parse_args()
    build_local_semantic_index(
        args.source,
        args.index_output,
        args.encoder_output,
        embeddings_output=args.embeddings_output,
        target_dimension=args.target_dimension,
    )


if __name__ == "__main__":
    main()
