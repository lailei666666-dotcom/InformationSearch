import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.semantic_retrieval.embedding_client import EmbeddingClient
from src.semantic_retrieval.embedding_cache import EmbeddingCache
from src.semantic_retrieval import runtime as semantic_runtime
from scripts.build_index import embed_reviews as embed_reviews_script

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBED_REVIEWS_SCRIPT = PROJECT_ROOT / "scripts" / "build_index" / "embed_reviews.py"
EMBEDDING_NAMESPACE = "openai_compatible:text-embedding-3-small:1536"


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.jsonl", namespace=EMBEDDING_NAMESPACE)

    cache.set("按键声音很小", [0.1, 0.2, 0.3])

    assert cache.get("按键声音很小") == [0.1, 0.2, 0.3]


def test_embedding_cache_persists_across_reloads_and_returns_copies(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    writer = EmbeddingCache(path, namespace=EMBEDDING_NAMESPACE)
    writer.set("按键声音很小", [0.1, 0.2, 0.3])

    reloaded = EmbeddingCache(path, namespace=EMBEDDING_NAMESPACE)
    vector = reloaded.get("按键声音很小")

    assert vector == [0.1, 0.2, 0.3]
    assert vector is not None
    vector.append(9.9)
    assert reloaded.get("按键声音很小") == [0.1, 0.2, 0.3]


def test_embedding_cache_separates_embedding_namespaces(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    keyboard_cache = EmbeddingCache(path, namespace="provider:model-a:3")
    lamp_cache = EmbeddingCache(path, namespace="provider:model-b:3")

    keyboard_cache.set("按键声音很小", [0.1, 0.2, 0.3])
    lamp_cache.set("按键声音很小", [0.7, 0.8, 0.9])

    assert EmbeddingCache(path, namespace="provider:model-a:3").get("按键声音很小") == [0.1, 0.2, 0.3]
    assert EmbeddingCache(path, namespace="provider:model-b:3").get("按键声音很小") == [0.7, 0.8, 0.9]


def test_embedding_cache_skips_corrupt_lines_during_reload(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "namespace": EMBEDDING_NAMESPACE,
                        "text": "按键声音很小",
                        "vector": [0.1, 0.2, 0.3],
                    },
                    ensure_ascii=False,
                ),
                '{"namespace":"broken"',
            ]
        ),
        encoding="utf-8",
    )

    cache = EmbeddingCache(path, namespace=EMBEDDING_NAMESPACE)

    assert cache.get("按键声音很小") == [0.1, 0.2, 0.3]


def test_embed_reviews_script_has_help_output() -> None:
    result = subprocess.run(
        [sys.executable, str(EMBED_REVIEWS_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0
    assert "--cache" in result.stdout


def test_build_default_client_embeds_uncached_reviews_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "reviews.csv"
    output = tmp_path / "embeddings.jsonl"
    pd.DataFrame([{"review_id": "R1", "clean_text": "按键声音很小"}]).to_csv(source, index=False)

    captured: dict[str, object] = {}

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        captured["texts"] = list(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(semantic_runtime, "build_openai_compatible_embedder", lambda **_: fake_embedder)

    client = embed_reviews_script.build_default_client(tmp_path / "cache.jsonl")
    records = embed_reviews_script.embed_reviews(source, output, client=client)

    assert captured["texts"] == ["按键声音很小"]
    assert records == [{"review_id": "R1", "text": "按键声音很小", "vector": [0.1, 0.2, 0.3]}]
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == records


def test_embedding_client_batches_only_uncached_texts(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.jsonl", namespace=EMBEDDING_NAMESPACE)
    cache.set("已缓存", [1.0, 1.1, 1.2])
    calls: list[list[str]] = []

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(index), float(index) + 0.1] for index, _ in enumerate(texts, start=1)]

    client = EmbeddingClient(embedder=fake_embedder, cache=cache, batch_size=2)
    vectors = client.embed_texts(["已缓存", "第一条", "第二条", "第三条"])

    assert calls == [["第一条", "第二条"], ["第三条"]]
    assert vectors == [[1.0, 1.1, 1.2], [1.0, 1.1], [2.0, 2.1], [1.0, 1.1]]


def test_embed_reviews_uses_batched_client_calls(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    output = tmp_path / "embeddings.jsonl"
    pd.DataFrame(
        [
            {"review_id": "R1", "clean_text": "第一条"},
            {"review_id": "R2", "clean_text": "第二条"},
            {"review_id": "R3", "clean_text": "第三条"},
        ]
    ).to_csv(source, index=False)
    calls: list[list[str]] = []

    class FakeClient:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

    records = embed_reviews_script.embed_reviews(source, output, client=FakeClient())

    assert calls == [["第一条", "第二条", "第三条"]]
    assert [record["review_id"] for record in records] == ["R1", "R2", "R3"]
