from dataclasses import dataclass
import re

from src.common.config import load_category_aliases


@dataclass(frozen=True)
class ParsedQuery:
    category: str | None
    need_text: str

def parse_query(text: str) -> ParsedQuery:
    cleaned_text = _clean_text(text)
    match = _find_best_match(cleaned_text)
    if match is None:
        return ParsedQuery(category=None, need_text=cleaned_text)

    category, alias, start = match
    need_text = _clean_text(cleaned_text[:start] + cleaned_text[start + len(alias) :])
    return ParsedQuery(category=category, need_text=need_text)


def _find_best_match(text: str) -> tuple[str, str, int] | None:
    best_match: tuple[str, str, int] | None = None
    for category, alias in load_category_aliases():
        start = text.find(alias)
        if start < 0:
            continue
        candidate = (category, alias, start)
        if best_match is None or _is_better_match(candidate, best_match):
            best_match = candidate
    return best_match


def _is_better_match(
    candidate: tuple[str, str, int],
    current: tuple[str, str, int],
) -> bool:
    _, candidate_alias, candidate_start = candidate
    _, current_alias, current_start = current
    return (-len(candidate_alias), candidate_start, candidate_alias) < (
        -len(current_alias),
        current_start,
        current_alias,
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
