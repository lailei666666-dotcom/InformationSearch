import pandas as pd
import pytest

from src.traditional_retrieval.bm25_engine import BM25Engine


def test_bm25_engine_ranks_relevant_keyboard_review_first() -> None:
    reviews = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "K1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "按键声音很小 图书馆用也不会影响别人",
            },
            {
                "review_id": "R2",
                "product_id": "L1",
                "product_name": "宿舍台灯A",
                "category": "台灯",
                "clean_text": "灯光柔和 晚上不刺眼",
            },
        ]
    )

    engine = BM25Engine.from_frame(reviews)
    results = engine.search("适合上课偷偷用的键盘", top_k=2)

    assert results[0]["review_id"] == "R1"


def test_bm25_engine_returns_no_hits_for_unmatched_query() -> None:
    reviews = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "K1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "按键声音很小 图书馆用也不会影响别人",
            }
        ]
    )

    engine = BM25Engine.from_frame(reviews)

    assert engine.search("防水登山靴", top_k=5) == []


def test_bm25_engine_from_frame_rejects_empty_required_fields() -> None:
    reviews = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "按键声音很小 图书馆用也不会影响别人",
            }
        ]
    )

    with pytest.raises(ValueError, match="product_id"):
        BM25Engine.from_frame(reviews)


def test_bm25_engine_save_load_round_trip_preserves_results(tmp_path) -> None:
    reviews = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "product_id": "K1",
                "product_name": "静音键盘A",
                "category": "键盘",
                "clean_text": "按键声音很小 图书馆用也不会影响别人",
            },
            {
                "review_id": "R2",
                "product_id": "L1",
                "product_name": "宿舍台灯A",
                "category": "台灯",
                "clean_text": "灯光柔和 晚上不刺眼",
            },
        ]
    )
    query = "适合上课偷偷用的键盘"

    engine = BM25Engine.from_frame(reviews)
    output_path = tmp_path / "bm25-index.json"
    engine.save(output_path)

    reloaded = BM25Engine.load(output_path)

    assert reloaded.search(query, top_k=2) == engine.search(query, top_k=2)
