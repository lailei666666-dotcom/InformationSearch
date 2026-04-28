import pytest

from scripts.run_retrieval.run_bm25 import run_bm25
from src.common.aggregation import aggregate_review_hits


def test_aggregate_review_hits_groups_reviews_into_product_results() -> None:
    hits = [
        {
            "review_id": "R1",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.9,
            "clean_text": "按键声音很小",
        },
        {
            "review_id": "R2",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.8,
            "clean_text": "宿舍晚上打字不吵",
        },
        {
            "review_id": "R3",
            "product_id": "K2",
            "product_name": "机械键盘B",
            "category": "键盘",
            "score": 0.6,
            "clean_text": "手感不错",
        },
    ]

    products = aggregate_review_hits(hits, product_top_n=2)

    assert products[0]["product_id"] == "K1"
    assert len(products[0]["evidence"]) == 2


def test_aggregate_review_hits_keeps_top_evidence_and_sorts_by_average_score() -> None:
    hits = [
        {
            "review_id": "R1",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.4,
            "clean_text": "基础评价",
        },
        {
            "review_id": "R2",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.9,
            "clean_text": "非常安静",
        },
        {
            "review_id": "R3",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.8,
            "clean_text": "夜里使用也不吵",
        },
        {
            "review_id": "R4",
            "product_id": "K2",
            "product_name": "机械键盘B",
            "category": "键盘",
            "score": 0.7,
            "clean_text": "手感不错",
        },
    ]

    products = aggregate_review_hits(hits, product_top_n=2, evidence_top_n=2)

    assert [evidence["review_id"] for evidence in products[0]["evidence"]] == ["R2", "R3"]
    assert products[0]["score"] == pytest.approx(0.7)
    assert products[1]["product_id"] == "K2"


def test_aggregate_review_hits_keeps_product_ranking_stable_when_evidence_count_changes() -> None:
    hits = [
        {
            "review_id": "R1",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.95,
            "clean_text": "最强单条评价",
        },
        {
            "review_id": "R2",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.10,
            "clean_text": "其余评价一般",
        },
        {
            "review_id": "R3",
            "product_id": "K2",
            "product_name": "机械键盘B",
            "category": "键盘",
            "score": 0.60,
            "clean_text": "整体稳定",
        },
        {
            "review_id": "R4",
            "product_id": "K2",
            "product_name": "机械键盘B",
            "category": "键盘",
            "score": 0.60,
            "clean_text": "评价比较一致",
        },
    ]

    one_evidence = aggregate_review_hits(hits, product_top_n=2, evidence_top_n=1)
    two_evidence = aggregate_review_hits(hits, product_top_n=2, evidence_top_n=2)

    assert [product["product_id"] for product in one_evidence] == ["K2", "K1"]
    assert [product["product_id"] for product in two_evidence] == ["K2", "K1"]
    assert len(one_evidence[0]["evidence"]) == 1
    assert len(two_evidence[0]["evidence"]) == 2


def test_run_bm25_aggregate_oversamples_review_hits_for_product_recall(monkeypatch, tmp_path) -> None:
    hits = [
        {
            "review_id": "R1",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.95,
            "clean_text": "第一条",
        },
        {
            "review_id": "R2",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.94,
            "clean_text": "第二条",
        },
        {
            "review_id": "R3",
            "product_id": "K1",
            "product_name": "静音键盘A",
            "category": "键盘",
            "score": 0.93,
            "clean_text": "第三条",
        },
        {
            "review_id": "R4",
            "product_id": "K2",
            "product_name": "机械键盘B",
            "category": "键盘",
            "score": 0.70,
            "clean_text": "另一款产品",
        },
    ]

    class FakeEngine:
        def __init__(self) -> None:
            self.requested_top_k: int | None = None

        def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
            self.requested_top_k = top_k
            return hits[:top_k]

    engine = FakeEngine()
    monkeypatch.setattr(
        "scripts.run_retrieval.run_bm25.BM25Engine.load",
        lambda index_path: engine,
    )

    products = run_bm25(
        tmp_path / "unused-index.json",
        "静音键盘",
        top_k=2,
        aggregate=True,
        product_top_n=2,
        evidence_top_n=1,
    )

    assert engine.requested_top_k is not None
    assert engine.requested_top_k > 2
    assert [product["product_id"] for product in products] == ["K1", "K2"]
