import json

from itertools import count

from src.common.xiaomi_reviews import (
    ProductSummary,
    XiaomiProduct,
    build_review_rows,
    collect_reviews,
    filter_catalog,
    parse_jsonp_payload,
)


def test_parse_jsonp_payload_extracts_json_object() -> None:
    payload = 'cb({"code":200,"msg":"success","data":{"value":1}});'

    parsed = parse_jsonp_payload(payload)

    assert parsed == {"code": 200, "msg": "success", "data": {"value": 1}}


def test_build_review_rows_creates_canonical_review_records() -> None:
    product = XiaomiProduct(category="耳机", product_id=20426, product_name="REDMI Buds 6")
    comments = [
        {
            "comment_id": "246980015",
            "user_name": "264*****05",
            "add_time": "2026-04-25",
            "total_grade": "5",
            "comment_content": "  外观很好看  \n  音质也不错 ",
            "up_num": 3,
            "user_reply_num": 1,
            "comment_images": ["https://example.com/1.jpg"],
        }
    ]

    rows = build_review_rows(product, comments)

    assert rows == [
        {
            "review_id": "mi-20426-246980015",
            "product_id": "20426",
            "product_name": "REDMI Buds 6",
            "category": "耳机",
            "raw_text": "外观很好看 音质也不错",
            "clean_text": "外观很好看 音质也不错",
            "rating": 5,
            "review_time": "2026-04-25",
            "source": "xiaomi_mall",
            "comment_id": "246980015",
            "user_name": "264*****05",
            "up_num": 3,
            "reply_count": 1,
            "has_images": True,
            "image_count": 1,
            "product_url": "https://www.mi.com/shop/comment/20426.html",
            "comment_url": "https://www.mi.com/comment/detail?comment_id=246980015",
        }
    ]


def test_filter_catalog_returns_requested_categories_only() -> None:
    products = filter_catalog(["键盘", "台灯"])

    assert products
    assert {product.category for product in products} == {"键盘", "台灯"}
    assert all(product.category != "耳机" for product in products)


def test_collect_reviews_handles_leading_empty_pages_before_real_results() -> None:
    product = XiaomiProduct(category="台灯", product_id=3806, product_name="米家 LED 智能台灯")
    comment_counter = count(1)

    def fake_summary_fetcher(*, product: XiaomiProduct, timeout: float) -> ProductSummary:
        return ProductSummary(
            comments_total=100,
            comments_good=99,
            default_good=90,
            satisfy_rate="99.0",
        )

    def fake_comment_fetcher(
        *,
        product: XiaomiProduct,
        page_index: int,
        page_size: int,
        session_id: str | None,
        timeout: float,
    ) -> dict[str, object]:
        if page_index < 3:
            return {"data": {"comments": []}}
        comment_id = str(next(comment_counter))
        return {
            "data": {
                "comments": [
                    {
                        "comment_id": comment_id,
                        "user_name": "用户A",
                        "add_time": "2026-04-27",
                        "total_grade": "5",
                        "comment_content": f"台灯评价 {comment_id}",
                        "up_num": 0,
                        "user_reply_num": 0,
                        "comment_images": [],
                    }
                ]
            }
        }

    rows, metadata = collect_reviews(
        products=[product],
        per_category_target=2,
        delay_seconds=0.0,
        summary_fetcher=fake_summary_fetcher,
        comment_page_fetcher=fake_comment_fetcher,
    )

    assert len(rows) == 2
    assert metadata["category_counts"] == {"台灯": 2}
    assert [row["review_id"] for row in rows] == ["mi-3806-1", "mi-3806-2"]
