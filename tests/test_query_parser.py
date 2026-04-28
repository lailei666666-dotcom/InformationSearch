from pathlib import Path

import pytest

from src.common import config as config_module
from src.common.config import load_category_aliases
from src.common.query_parser import parse_query


def test_parse_query_extracts_category_and_need_text() -> None:
    parsed = parse_query("适合上课偷偷用的键盘")

    assert parsed.category == "键盘"
    assert parsed.need_text == "适合上课偷偷用的"


def test_parse_query_returns_original_text_when_no_category_detected() -> None:
    parsed = parse_query("适合上课偷偷用的")

    assert parsed.category is None
    assert parsed.need_text == "适合上课偷偷用的"


def test_parse_query_prefers_longest_matching_alias() -> None:
    parsed = parse_query("适合宿舍用的蓝牙键盘")

    assert parsed.category == "键盘"
    assert parsed.need_text == "适合宿舍用的"


def test_parse_query_works_with_only_categories_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "categories.yml").write_text(
        "categories:\n  键盘:\n    - 键盘\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)
    load_category_aliases.cache_clear()

    parsed = parse_query("安静点的键盘")

    assert parsed.category == "键盘"
    assert parsed.need_text == "安静点的"
