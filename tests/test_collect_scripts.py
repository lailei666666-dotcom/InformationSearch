from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "collect" / "download_public_dataset.py"
SCRAPE_SCRIPT = PROJECT_ROOT / "scripts" / "collect" / "scrape_reviews_demo.py"
XIAOMI_SCRAPE_SCRIPT = PROJECT_ROOT / "scripts" / "collect" / "scrape_xiaomi_reviews.py"


def test_download_script_has_help_output() -> None:
    result = subprocess.run(
        [sys.executable, str(DOWNLOAD_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "--output" in result.stdout


def test_download_script_prepares_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"

    result = subprocess.run(
        [sys.executable, str(DOWNLOAD_SCRIPT), "--output", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert output_dir.is_dir()


def test_scrape_reviews_demo_has_help_output() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRAPE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "--category" in result.stdout
    assert "--output" in result.stdout


def test_scrape_reviews_demo_writes_csv_template(tmp_path: Path) -> None:
    output_path = tmp_path / "demo.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRAPE_SCRIPT),
            "--category",
            "keyboards",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["category", "review_id", "rating", "title", "content"],
        ["keyboards", "", "", "", ""],
    ]


def test_scrape_xiaomi_reviews_has_help_output() -> None:
    result = subprocess.run(
        [sys.executable, str(XIAOMI_SCRAPE_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "--per-category-target" in result.stdout
    assert "--dry-run" in result.stdout


def test_scrape_xiaomi_reviews_dry_run_lists_catalog() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(XIAOMI_SCRAPE_SCRIPT),
            "--category",
            "耳机",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload
    assert all(item["category"] == "耳机" for item in payload)
    assert all(item["product_id"] for item in payload)
