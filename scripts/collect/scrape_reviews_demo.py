from __future__ import annotations

import sys
import csv
from pathlib import Path

import typer


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.paths import DATA_DIR  # noqa: E402


def main(
    category: str = typer.Option(..., "--category", help="Product category label."),
    output: Path = typer.Option(
        DATA_DIR / "raw" / "reviews_demo.csv",
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="CSV path for the demo review template.",
    ),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "review_id", "rating", "title", "content"])
        writer.writerow([category, "", "", "", ""])
    typer.echo(f"Prepared demo review template: {output}")


if __name__ == "__main__":
    typer.run(main)
