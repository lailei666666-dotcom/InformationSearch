from pathlib import Path

import pandas as pd

from src.common.schemas import ReviewRecord
from src.common.text import normalize_label, normalize_whitespace

CANONICAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "review_id": ("review_id", "评论编号", "评论id", "评价编号", "id"),
    "product_id": ("product_id", "商品编号", "商品id", "sku", "item_id"),
    "product_name": ("product_name", "商品标题", "商品名称", "产品名称", "title"),
    "source_category": ("source_category", "原始类目", "类目", "分类", "category"),
    "raw_text": ("raw_text", "评论内容", "评价内容", "内容", "review_text"),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "键盘": ("键盘", "机械键盘", "办公键盘", "游戏键盘"),
    "台灯": ("台灯", "护眼灯", "桌面台灯", "阅读灯"),
    "耳机": ("耳机", "蓝牙耳机", "有线耳机", "蓝牙耳麦", "耳麦"),
    "笔记本": ("笔记本", "笔记本电脑", "轻薄本", "办公本"),
    "充电宝": ("充电宝", "移动电源", "快充电源", "便携电源"),
}

CATEGORY_ALIAS_LOOKUP: dict[str, str] = {
    normalize_label(alias): canonical
    for canonical, aliases in CATEGORY_ALIASES.items()
    for alias in aliases
}


def normalize_reviews(source: Path, output: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    renamed = frame.rename(columns=_build_column_rename_map(frame.columns))
    _require_columns(renamed, CANONICAL_COLUMNS.keys())

    normalized = renamed.loc[:, list(CANONICAL_COLUMNS.keys())].copy()
    normalized["review_id"] = normalized["review_id"].map(_normalize_required_text)
    normalized["product_id"] = normalized["product_id"].map(_normalize_required_text)
    normalized["product_name"] = normalized["product_name"].map(_normalize_required_text)
    normalized["raw_text"] = normalized["raw_text"].map(_normalize_required_text)
    normalized["category"] = normalized["source_category"].map(_map_category)
    normalized["clean_text"] = normalized["raw_text"].map(normalize_whitespace)
    normalized = normalized.loc[
        :,
        ["review_id", "product_id", "product_name", "category", "raw_text", "clean_text"],
    ]

    records = [
        ReviewRecord.model_validate(record).model_dump()
        for record in normalized.to_dict("records")
    ]
    result = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def _build_column_rename_map(columns: pd.Index) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in columns:
                rename_map[alias] = canonical
                break
    return rename_map


def _require_columns(frame: pd.DataFrame, required_columns: object) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Missing required review columns: {missing_display}")


def _normalize_required_text(value: object) -> str:
    normalized = normalize_whitespace("" if pd.isna(value) else str(value))
    if not normalized:
        raise ValueError("Normalized review data contains an empty required field")
    return normalized


def _map_category(value: object) -> str:
    normalized = _normalize_required_text(value)
    normalized_label = normalize_label(normalized)
    canonical = CATEGORY_ALIAS_LOOKUP.get(normalized_label)
    if canonical is None:
        raise ValueError(f"Unsupported category value: {normalized}")
    return canonical


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize review data into the canonical schema."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    normalize_reviews(args.source, args.output)


if __name__ == "__main__":
    main()
