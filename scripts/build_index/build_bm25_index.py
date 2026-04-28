from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.io_utils import read_table
from src.traditional_retrieval.bm25_engine import BM25Engine


def build_bm25_index(source: Path, output: Path) -> BM25Engine:
    frame = read_table(source)
    engine = BM25Engine.from_frame(frame)
    engine.save(output)
    return engine


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a small serialized BM25 review index.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_bm25_index(args.source, args.output)


if __name__ == "__main__":
    main()
