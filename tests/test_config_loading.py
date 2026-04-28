from pathlib import Path

import pytest

from src.common import config as config_module
from src.common.config import load_category_aliases, load_settings


def test_load_settings_returns_expected_categories() -> None:
    settings = load_settings()

    assert "键盘" in settings.categories
    assert "台灯" in settings.categories
    assert settings.embedding.provider
    assert settings.experiment.top_k == 10


def test_load_settings_requires_categories_top_level_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "categories.yml",
        "wrong_key:\n  键盘:\n    - 键盘\n",
    )
    _write_config(
        tmp_path / "embedding.yml",
        "provider: openai_compatible\nmodel: text-embedding-3-small\nbatch_size: 32\ndimensions: 1536\n",
    )
    _write_config(
        tmp_path / "experiment.yml",
        "top_k: 10\nproduct_top_n: 3\nalpha: 0.65\n",
    )
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)

    with pytest.raises(ValueError, match="categories"):
        load_settings()


def test_load_settings_rejects_unknown_embedding_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "categories.yml",
        "categories:\n  键盘:\n    - 键盘\n",
    )
    _write_config(
        tmp_path / "embedding.yml",
        "provider: openai_compatible\nmodel: text-embedding-3-small\nbatch_size: 32\ndimensions: 1536\nunexpected: true\n",
    )
    _write_config(
        tmp_path / "experiment.yml",
        "top_k: 10\nproduct_top_n: 3\nalpha: 0.65\n",
    )
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)

    with pytest.raises(Exception, match="unexpected"):
        load_settings()


def test_load_category_aliases_rejects_duplicate_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path / "categories.yml",
        (
            "categories:\n"
            "  键盘:\n"
            "    - 通用\n"
            "  台灯:\n"
            "    - 通用\n"
        ),
    )
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path)
    load_category_aliases.cache_clear()

    with pytest.raises(ValueError, match="duplicate alias"):
        load_category_aliases()


def _write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
