from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
import typer


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

from src.common.paths import DATA_DIR  # noqa: E402
from src.common.xiaomi_reviews import (  # noqa: E402
    XIAOMI_PRODUCT_CATALOG,
    collect_reviews,
    filter_catalog,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Path = typer.Option(
        DATA_DIR / "raw" / "xiaomi_reviews.csv",
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="CSV path for collected Xiaomi Mall reviews.",
    ),
    summary_output: Path = typer.Option(
        DATA_DIR / "raw" / "xiaomi_reviews_summary.json",
        "--summary-output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="JSON path for collection metadata and product summaries.",
    ),
    category: list[str] | None = typer.Option(
        None,
        "--category",
        help="Restrict collection to one or more categories. Repeat the flag to pass multiple values.",
    ),
    per_category_target: int = typer.Option(
        2500,
        "--per-category-target",
        min=1,
        help="Target number of collected reviews per category.",
    ),
    page_size: int = typer.Option(
        20,
        "--page-size",
        min=1,
        max=20,
        help="Requested Xiaomi API page size. The public endpoint caps this at 20.",
    ),
    delay_seconds: float = typer.Option(
        0.05,
        "--delay-seconds",
        min=0.0,
        help="Delay between page requests so the crawler behaves politely.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the selected product catalog without collecting any reviews.",
    ),
) -> None:
    catalog = filter_catalog(category)
    if not catalog:
        available = ", ".join(sorted({product.category for product in XIAOMI_PRODUCT_CATALOG}))
        raise typer.BadParameter(f"No Xiaomi catalog entries match the requested categories. Available: {available}")

    if dry_run:
        payload = [
            {
                "category": product.category,
                "product_id": product.product_id,
                "product_name": product.product_name,
                "comment_page_url": product.comment_page_url,
            }
            for product in catalog
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    rows, summary = collect_reviews(
        products=catalog,
        per_category_target=per_category_target,
        page_size=page_size,
        delay_seconds=delay_seconds,
    )

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    category_counts = summary["category_counts"]
    typer.echo(
        "Collected "
        f"{summary['collected_total']} Xiaomi Mall reviews across "
        f"{len(category_counts)} categories -> {output}"
    )


if __name__ == "__main__":
    app()
