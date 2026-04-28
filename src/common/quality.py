import re

import pandas as pd

from src.common.text import normalize_whitespace


_MEANINGFUL_CHAR_RE = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)
_LANGUAGE_SIGNAL_RE = re.compile(r"[A-Za-z\u4e00-\u9fff]", re.UNICODE)
DEFAULT_MIN_TEXT_LENGTH = 6


def normalize_review_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return normalize_whitespace(str(value))


def has_minimum_review_text(text: str, min_length: int = DEFAULT_MIN_TEXT_LENGTH) -> bool:
    meaningful_length = len(_MEANINGFUL_CHAR_RE.findall(text))
    return meaningful_length >= min_length and bool(_LANGUAGE_SIGNAL_RE.search(text))


def low_quality_text_mask(
    values: pd.Series, min_length: int = DEFAULT_MIN_TEXT_LENGTH
) -> pd.Series:
    normalized = values.map(normalize_review_text)
    return ~normalized.map(lambda text: has_minimum_review_text(text, min_length=min_length))
