from __future__ import annotations

import sys
from pathlib import Path
import shutil

def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.evaluation.qrels import ANNOTATIONS_DIR


def bootstrap_annotation_assets(output_dir: Path = ANNOTATIONS_DIR, force: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = ("queries.csv", "qrels_reviews.csv", "qrels_products.csv")
    written_paths: list[Path] = []

    for filename in assets:
        source_path = ANNOTATIONS_DIR / filename
        path = output_dir / filename
        if path.exists() and not force:
            continue
        if path.exists() and path.resolve() == source_path.resolve():
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Bootstrap annotation source file not found: {source_path}")

        shutil.copyfile(source_path, path)
        written_paths.append(path)

    return written_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Initialize bootstrap annotation CSV files.")
    parser.add_argument("--output-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files.")
    args = parser.parse_args()

    written_paths = bootstrap_annotation_assets(output_dir=args.output_dir, force=args.force)
    if not written_paths:
        print("Annotation assets already exist.")
        return

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
