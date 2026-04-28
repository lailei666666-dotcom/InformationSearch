from __future__ import annotations

import sys
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
    output: Path = typer.Option(
        DATA_DIR / "raw" / "public_dataset",
        "--output",
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where the downloaded public dataset will be staged.",
    ),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Prepared dataset directory: {output}")


if __name__ == "__main__":
    typer.run(main)
