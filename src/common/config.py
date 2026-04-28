from functools import lru_cache

from src.common.io_utils import load_yaml
from src.common.paths import CONFIGS_DIR
from src.common.schemas import CategoriesSettings, Settings


def load_categories() -> dict[str, list[str]]:
    categories_payload = load_yaml(CONFIGS_DIR / "categories.yml")
    if "categories" not in categories_payload:
        raise ValueError("categories.yml must contain a top-level 'categories' key")
    return CategoriesSettings.model_validate(categories_payload).categories


@lru_cache(maxsize=1)
def load_category_aliases() -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    seen_aliases: dict[str, str] = {}
    for category, category_aliases in load_categories().items():
        for alias in category_aliases:
            existing_category = seen_aliases.get(alias)
            if existing_category is not None and existing_category != category:
                raise ValueError(
                    f"Found duplicate alias '{alias}' for categories "
                    f"'{existing_category}' and '{category}'"
                )
            seen_aliases[alias] = category
            aliases.append((category, alias))
    return tuple(aliases)


def load_settings() -> Settings:
    embedding = load_yaml(CONFIGS_DIR / "embedding.yml")
    experiment = load_yaml(CONFIGS_DIR / "experiment.yml")
    return Settings(
        categories=load_categories(),
        embedding=embedding,
        experiment=experiment,
    )
