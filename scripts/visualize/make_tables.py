from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "system",
    "target",
    "k",
    "min_relevance",
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "source_file",
]


def flatten_benchmark_summary(benchmark_path: Path, *, system: str | None = None) -> dict[str, Any]:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8-sig"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Benchmark file is missing a summary object: {benchmark_path}")

    return {
        "system": system or benchmark_path.stem,
        "target": payload.get("target", ""),
        "k": payload.get("k", ""),
        "min_relevance": payload.get("min_relevance", ""),
        "precision_at_k": summary.get("precision_at_k", ""),
        "recall_at_k": summary.get("recall_at_k", ""),
        "mrr": summary.get("mrr", ""),
        "source_file": str(benchmark_path),
    }


def write_summary_table(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise ValueError("Summary table rows cannot be empty.")

    fieldnames = _resolve_fieldnames(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(
    benchmark_paths: list[Path],
    *,
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    label_overrides = labels or {}
    return [
        flatten_benchmark_summary(path, system=_resolve_system_label(path, label_overrides))
        for path in benchmark_paths
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a summary CSV from benchmark result JSON files.")
    parser.add_argument("benchmark_paths", nargs="+", type=Path, help="Benchmark result JSON file(s).")
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV path.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="PATH=SYSTEM",
        help="Optional system label override for a benchmark path.",
    )
    return parser


def _parse_label_overrides(raw_labels: list[str]) -> dict[Path, str]:
    overrides: dict[str, str] = {}
    for item in raw_labels:
        path_text, separator, system = item.partition("=")
        if not separator or not system:
            raise ValueError(f"Invalid label override: {item}")
        overrides[path_text] = system
    return overrides


def _resolve_system_label(benchmark_path: Path, labels: dict[str, str]) -> str | None:
    for key in (str(benchmark_path), benchmark_path.name, benchmark_path.stem):
        if key in labels:
            return labels[key]
    return None


def _resolve_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    row_keys = {key for row in rows for key in row}
    ordered = [column for column in SUMMARY_COLUMNS if column in row_keys]
    extras = sorted(row_keys - set(ordered))
    return ordered + extras


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    label_overrides = _parse_label_overrides(args.label)
    rows = build_summary_rows(args.benchmark_paths, labels=label_overrides)
    write_summary_table(rows, args.output)


if __name__ == "__main__":
    main()
