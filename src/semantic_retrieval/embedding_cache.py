from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path


class EmbeddingCache:
    """Small persistent text-to-vector cache backed by JSONL."""

    def __init__(self, path: Path, *, namespace: str = "default") -> None:
        self.path = path
        self.namespace = namespace
        self._entries: dict[str, list[float]] = {}
        self._load()

    def get(self, text: str) -> list[float] | None:
        vector = self._entries.get(text)
        if vector is None:
            return None
        return list(vector)

    def set(self, text: str, vector: list[float]) -> list[float]:
        existing = self._entries.get(text)
        if existing == vector:
            return list(vector)

        normalized = [float(value) for value in vector]
        record = {"namespace": self.namespace, "text": text, "vector": normalized}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._entries[text] = normalized
        return list(normalized)

    def _load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = self._parse_record(raw)
                if record is None:
                    continue
                if record["namespace"] != self.namespace:
                    continue
                self._entries[record["text"]] = record["vector"]

    def _parse_record(self, raw: str) -> dict[str, str | list[float]] | None:
        try:
            record = json.loads(raw)
        except JSONDecodeError:
            return None

        if not isinstance(record, dict):
            return None

        text = record.get("text")
        vector = record.get("vector")
        namespace = record.get("namespace", "default")
        if not isinstance(text, str) or not isinstance(namespace, str) or not isinstance(vector, list):
            return None

        try:
            normalized_vector = [float(value) for value in vector]
        except (TypeError, ValueError):
            return None

        return {"namespace": namespace, "text": text, "vector": normalized_vector}
