from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.io_utils import read_table  # noqa: E402
from src.semantic_retrieval.embedding_client import EmbeddingClient  # noqa: E402
from src.semantic_retrieval.runtime import build_default_embedding_client  # noqa: E402

REQUIRED_COLUMNS = ("review_id",)
TEXT_COLUMNS = ("clean_text", "raw_text")


def embed_reviews(
    source: Path,
    output: Path,
    *,
    client: EmbeddingClient,
) -> list[dict[str, object]]:
    frame = read_table(source)
    _require_columns(frame, REQUIRED_COLUMNS)

    text_column = _select_text_column(frame)
    pending_rows: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        text = str(getattr(row, text_column, "")).strip()
        if not text:
            continue
        pending_rows.append((str(row.review_id), text))

    vectors = client.embed_texts([text for _, text in pending_rows])
    records = [
        {"review_id": review_id, "text": text, "vector": vector}
        for (review_id, text), vector in zip(pending_rows, vectors, strict=True)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def build_default_client(cache_path: Path) -> EmbeddingClient:
    return build_default_embedding_client(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed review text with a persistent local cache.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs") / "semantic" / "embedding-cache.jsonl",
        help="JSONL cache path for text embeddings.",
    )
    args = parser.parse_args()
    embed_reviews(args.source, args.output, client=build_default_client(args.cache))


def _require_columns(frame, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def _select_text_column(frame: pd.DataFrame) -> str:
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError("Missing required review text column: clean_text or raw_text")
if __name__ == "__main__":
    main()
