from pathlib import Path


def test_repository_layout_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/annotations",
        "outputs/indexes",
        "outputs/runs",
        "outputs/figures",
        "outputs/tables",
        "src/common",
        "src/traditional_retrieval",
        "src/semantic_retrieval",
        "src/hybrid_retrieval",
        "src/evaluation",
        "scripts/collect",
        "scripts/preprocess",
        "scripts/build_index",
        "scripts/run_retrieval",
        "scripts/evaluate",
        "scripts/visualize",
    ]
    missing = [item for item in expected if not (root / item).exists()]
    assert not missing, missing
