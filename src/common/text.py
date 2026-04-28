import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_label(value: str) -> str:
    normalized = normalize_whitespace(value)
    return normalized.casefold()
