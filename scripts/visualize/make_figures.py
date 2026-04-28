from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def write_metric_bar_chart(summary_csv: Path, output: Path, *, metric: str = "mrr") -> None:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Summary CSV has no rows: {summary_csv}")
    if metric not in rows[0]:
        raise ValueError(f"Metric column not found in summary CSV: {metric}")

    systems = [row["system"] for row in rows]
    values = [float(row[metric]) for row in rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(systems, values, color="#4C78A8")
    ax.set_xlabel("System")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by system")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create bar chart figures from a benchmark summary CSV.")
    parser.add_argument("summary_csv", type=Path, help="Input summary CSV file.")
    parser.add_argument("--output", required=True, type=Path, help="Destination image path.")
    parser.add_argument("--metric", default="mrr", help="Summary metric column to plot.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    write_metric_bar_chart(args.summary_csv, args.output, metric=args.metric)


if __name__ == "__main__":
    main()
