from __future__ import annotations

import re

from src.common.text import normalize_whitespace


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    tokens: list[str] = []
    for chunk in _TOKEN_RE.findall(normalized.casefold()):
        if _is_cjk(chunk):
            tokens.extend(_tokenize_cjk(chunk))
            continue
        tokens.append(chunk)
    return tokens


def _is_cjk(value: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in value)


def _tokenize_cjk(value: str) -> list[str]:
    if len(value) == 1:
        return [value]

    tokens = [value]
    tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
    return tokens
