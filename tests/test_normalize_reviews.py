from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess.normalize_reviews import normalize_reviews


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_normalize_reviews_maps_columns_and_categories(tmp_path: Path) -> None:
    source = FIXTURES_DIR / "sample_reviews.csv"
    output = tmp_path / "normalized.parquet"

    frame = normalize_reviews(source, output)

    expected_columns = [
        "review_id",
        "product_id",
        "product_name",
        "category",
        "raw_text",
        "clean_text",
    ]
    assert frame.columns.tolist() == expected_columns
    assert frame["category"].tolist() == ["键盘", "台灯", "耳机", "笔记本", "充电宝"]
    assert frame.loc[0, "clean_text"] == "手感很好，回弹清晰。"
    assert output.exists()


def test_normalize_reviews_rejects_unsupported_category(tmp_path: Path) -> None:
    source = tmp_path / "unsupported.csv"
    source.write_text(
        "评论编号,商品编号,商品标题,原始类目,评论内容\n"
        "R-1,P-1,桌面音箱,音箱,声音通透。\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported category value: 音箱"):
        normalize_reviews(source, tmp_path / "normalized.parquet")


def test_normalize_reviews_requires_exact_category_alias_match(tmp_path: Path) -> None:
    source = tmp_path / "inexact_category.csv"
    source.write_text(
        "评论编号,商品编号,商品标题,原始类目,评论内容\n"
        "R-1,P-1,机械键盘套装,机械键盘套装,组合很方便。\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported category value: 机械键盘套装"):
        normalize_reviews(source, tmp_path / "normalized.parquet")


def test_normalize_reviews_requires_expected_columns(tmp_path: Path) -> None:
    source = tmp_path / "missing_columns.csv"
    source.write_text(
        "评论编号,商品编号,商品标题,评论内容\n"
        "R-1,P-1,玄轴机械键盘,手感很好。\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required review columns: source_category"):
        normalize_reviews(source, tmp_path / "normalized.parquet")


def test_normalize_reviews_normalizes_clean_text_whitespace(tmp_path: Path) -> None:
    source = tmp_path / "whitespace.csv"
    source.write_text(
        "评论编号,商品编号,商品标题,原始类目,评论内容\n"
        'R-1,P-1,玄轴机械键盘,机械键盘,"  手感很好  \n  回弹清晰\t "\n',
        encoding="utf-8",
    )

    frame = normalize_reviews(source, tmp_path / "normalized.parquet")

    assert frame.loc[0, "raw_text"] == "手感很好 回弹清晰"
    assert frame.loc[0, "clean_text"] == "手感很好 回弹清晰"


def test_normalize_reviews_writes_explicit_output_contract(tmp_path: Path) -> None:
    source = FIXTURES_DIR / "sample_reviews.csv"
    output = tmp_path / "normalized.parquet"

    frame = normalize_reviews(source, output)
    persisted = pd.read_parquet(output)

    expected = pd.DataFrame(
        [
            {
                "review_id": "R-1001",
                "product_id": "P-001",
                "product_name": "玄轴机械键盘",
                "category": "键盘",
                "raw_text": "手感很好，回弹清晰。",
                "clean_text": "手感很好，回弹清晰。",
            },
            {
                "review_id": "R-1002",
                "product_id": "P-002",
                "product_name": "书桌护眼灯",
                "category": "台灯",
                "raw_text": "亮度均匀，晚上看书很舒服。",
                "clean_text": "亮度均匀，晚上看书很舒服。",
            },
            {
                "review_id": "R-1003",
                "product_id": "P-003",
                "product_name": "降噪蓝牙耳机",
                "category": "耳机",
                "raw_text": "地铁里也能听清，人声突出。",
                "clean_text": "地铁里也能听清，人声突出。",
            },
            {
                "review_id": "R-1004",
                "product_id": "P-004",
                "product_name": "轻薄办公本",
                "category": "笔记本",
                "raw_text": "续航扎实，办公够用。",
                "clean_text": "续航扎实，办公够用。",
            },
            {
                "review_id": "R-1005",
                "product_id": "P-005",
                "product_name": "22W快充电源",
                "category": "充电宝",
                "raw_text": "充电速度快，出门携带方便。",
                "clean_text": "充电速度快，出门携带方便。",
            },
        ]
    )

    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(persisted.reset_index(drop=True), expected)
