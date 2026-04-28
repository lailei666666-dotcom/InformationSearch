from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from src.common.text import normalize_whitespace

SUMMARY_ENDPOINT = "https://api2.service.order.mi.com/user_comment/get_summary"
LIST_ENDPOINT = "https://api2.service.order.mi.com/user_comment/get_list"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)
MAX_PAGE_SIZE = 20


@dataclass(frozen=True)
class XiaomiProduct:
    category: str
    product_id: int
    product_name: str

    @property
    def comment_page_url(self) -> str:
        return f"https://www.mi.com/shop/comment/{self.product_id}.html"


@dataclass(frozen=True)
class ProductSummary:
    comments_total: int
    comments_good: int
    default_good: int
    satisfy_rate: str


XIAOMI_PRODUCT_CATALOG: tuple[XiaomiProduct, ...] = (
    XiaomiProduct(category="键盘", product_id=19380, product_name="小米便携双模键盘"),
    XiaomiProduct(category="键盘", product_id=8133, product_name="小米游戏键盘"),
    XiaomiProduct(category="键盘", product_id=18336, product_name="Xiaomi Pad 6/6 Pro 智能触控键盘"),
    XiaomiProduct(category="键盘", product_id=19668, product_name="Xiaomi Pad 6S Pro 智能触控键盘"),
    XiaomiProduct(category="键盘", product_id=20594, product_name="Xiaomi Pad 7/7 Pro 键盘式双面保护壳"),
    XiaomiProduct(category="键盘", product_id=16353, product_name="小米平板 键盘式双面保护壳 12.4"),
    XiaomiProduct(category="台灯", product_id=3806, product_name="米家 LED 智能台灯"),
    XiaomiProduct(category="台灯", product_id=9743, product_name="米家台灯1S"),
    XiaomiProduct(category="台灯", product_id=14825, product_name="米家智能台灯Lite"),
    XiaomiProduct(category="台灯", product_id=18377, product_name="米家多功能充电台灯"),
    XiaomiProduct(category="台灯", product_id=16648, product_name="米家台灯1S增强版 耀夜黑"),
    XiaomiProduct(category="台灯", product_id=19858, product_name="米家台灯2"),
    XiaomiProduct(category="台灯", product_id=20565, product_name="米家桌面学习灯"),
    XiaomiProduct(category="台灯", product_id=16674, product_name="米家台灯Pro 读写版"),
    XiaomiProduct(category="台灯", product_id=19927, product_name="米家台灯2 Lite"),
    XiaomiProduct(category="耳机", product_id=20426, product_name="REDMI Buds 6"),
    XiaomiProduct(category="耳机", product_id=19180, product_name="Redmi Buds 5"),
    XiaomiProduct(category="耳机", product_id=20007, product_name="REDMI Buds 6 活力版"),
    XiaomiProduct(category="耳机", product_id=4792, product_name="小米圈铁耳机Pro"),
    XiaomiProduct(category="笔记本", product_id=10000179, product_name="RedmiBook 14 增强版"),
    XiaomiProduct(category="笔记本", product_id=10000153, product_name="RedmiBook 14 独显版"),
    XiaomiProduct(category="笔记本", product_id=10000170, product_name="RedmiBook 14"),
    XiaomiProduct(category="笔记本", product_id=10000113, product_name="小米游戏本"),
    XiaomiProduct(category="笔记本", product_id=10000281, product_name="RedmiBook Pro 15"),
    XiaomiProduct(category="笔记本", product_id=19715, product_name="Redmi Book Pro 14"),
    XiaomiProduct(category="笔记本", product_id=14125, product_name="RedmiBook Pro 14 i5-H/锐炬Xe/灰"),
    XiaomiProduct(category="笔记本", product_id=10000242, product_name="RedmiBook 16"),
    XiaomiProduct(category="笔记本", product_id=10000226, product_name="RedmiBook 16 锐龙版"),
    XiaomiProduct(category="笔记本", product_id=12445, product_name="RedmiBook 16 i7 16G 512G MX350 灰"),
    XiaomiProduct(category="笔记本", product_id=10050037, product_name="Redmi Book 15E"),
    XiaomiProduct(category="笔记本", product_id=10000032, product_name='小米笔记本Air 12.5"'),
    XiaomiProduct(category="充电宝", product_id=4580, product_name="小米移动电源2"),
    XiaomiProduct(category="充电宝", product_id=11174, product_name="小米移动电源3 10000mAh 快充版"),
    XiaomiProduct(category="充电宝", product_id=7607, product_name="移动电源2 (5000mAh)"),
    XiaomiProduct(category="充电宝", product_id=7463, product_name="移动电源2（10000mAh）"),
    XiaomiProduct(category="充电宝", product_id=6802, product_name="移动电源2C（20000mAh）"),
    XiaomiProduct(category="充电宝", product_id=4998, product_name="移动电源高配版（10000mAh）"),
)

_JSONP_RE = re.compile(r"^[^(]+\((.*)\);?$", re.DOTALL)


def parse_jsonp_payload(payload: str) -> dict[str, Any]:
    match = _JSONP_RE.match(payload.strip())
    if not match:
        raise ValueError("Expected JSONP payload")
    return json.loads(match.group(1))


def fetch_summary(product: XiaomiProduct, timeout: float = 20.0) -> ProductSummary:
    data = _request_jsonp(
        SUMMARY_ENDPOINT,
        params={
            "show_all_tag": 1,
            "goods_id": product.product_id,
            "v_pid": product.product_id,
            "support_start": 0,
            "support_len": 10,
            "add_start": 0,
            "add_len": 10,
            "profile_id": 0,
            "show_img": 0,
        },
        referer=product.comment_page_url,
        timeout=timeout,
    )
    detail = data.get("data", {}).get("detail", {})
    return ProductSummary(
        comments_total=_safe_int(detail.get("comments_total")),
        comments_good=_safe_int(detail.get("comments_good")),
        default_good=_safe_int(detail.get("default_good")),
        satisfy_rate=str(detail.get("satisfy_per", "")),
    )


def fetch_comment_page(
    product: XiaomiProduct,
    page_index: int,
    page_size: int = MAX_PAGE_SIZE,
    order_by: int = 21,
    session_id: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    bounded_page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    return _request_jsonp(
        LIST_ENDPOINT,
        params={
            "goods_id": product.product_id,
            "v_pid": product.product_id,
            "order_by": order_by,
            "page_index": page_index,
            "page_size": bounded_page_size,
            "profile_id": 0,
            "show_img": 0,
            "session_id": session_id or str(uuid4()),
        },
        referer=product.comment_page_url,
        timeout=timeout,
    )


def build_review_rows(product: XiaomiProduct, comments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comment in comments:
        content = normalize_whitespace(str(comment.get("comment_content") or ""))
        if not content:
            continue
        comment_id = str(comment.get("comment_id") or "").strip()
        if not comment_id:
            continue
        rating = _safe_int(comment.get("total_grade"))
        comment_images = comment.get("comment_images") or []
        row = {
            "review_id": f"mi-{product.product_id}-{comment_id}",
            "product_id": str(product.product_id),
            "product_name": product.product_name,
            "category": product.category,
            "raw_text": content,
            "clean_text": content,
            "rating": rating,
            "review_time": str(comment.get("add_time") or "").strip(),
            "source": "xiaomi_mall",
            "comment_id": comment_id,
            "user_name": str(comment.get("user_name") or "").strip(),
            "up_num": _safe_int(comment.get("up_num")),
            "reply_count": _safe_int(comment.get("user_reply_num")),
            "has_images": bool(comment_images),
            "image_count": len(comment_images),
            "product_url": product.comment_page_url,
            "comment_url": f"https://www.mi.com/comment/detail?comment_id={comment_id}",
        }
        rows.append(row)
    return rows


def collect_reviews(
    products: Iterable[XiaomiProduct],
    per_category_target: int,
    page_size: int = MAX_PAGE_SIZE,
    delay_seconds: float = 0.0,
    timeout: float = 20.0,
    max_leading_empty_pages: int = 8,
    max_consecutive_empty_pages: int = 3,
    max_pages_per_product: int = 500,
    summary_fetcher: Any = fetch_summary,
    comment_page_fetcher: Any = fetch_comment_page,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = list(products)
    if per_category_target <= 0:
        raise ValueError("per_category_target must be positive")

    grouped = group_products_by_category(catalog)
    summaries = {
        product.product_id: summary_fetcher(product=product, timeout=timeout)
        for product in catalog
    }
    page_positions = {product.product_id: 1 for product in catalog}
    session_ids = {product.product_id: str(uuid4()) for product in catalog}
    empty_page_counts = {product.product_id: 0 for product in catalog}
    seen_non_empty_page = {product.product_id: False for product in catalog}
    exhausted: set[int] = set()
    seen_review_ids: set[str] = set()
    collected_rows: list[dict[str, Any]] = []
    category_counts = {category: 0 for category in grouped}

    for category, category_products in grouped.items():
        while category_counts[category] < per_category_target:
            progressed = False
            for product in category_products:
                if product.product_id in exhausted:
                    continue

                if page_positions[product.product_id] > max_pages_per_product:
                    exhausted.add(product.product_id)
                    continue

                payload = comment_page_fetcher(
                    product=product,
                    page_index=page_positions[product.product_id],
                    page_size=page_size,
                    session_id=session_ids[product.product_id],
                    timeout=timeout,
                )
                data = payload.get("data", {})
                comments = data.get("comments") or []

                if not comments:
                    empty_page_counts[product.product_id] += 1
                    limit = (
                        max_consecutive_empty_pages
                        if seen_non_empty_page[product.product_id]
                        else max_leading_empty_pages
                    )
                    page_positions[product.product_id] += 1
                    progressed = True
                    if empty_page_counts[product.product_id] >= limit:
                        exhausted.add(product.product_id)
                    continue

                seen_non_empty_page[product.product_id] = True
                empty_page_counts[product.product_id] = 0
                progressed = True
                for row in build_review_rows(product, comments):
                    if row["review_id"] in seen_review_ids:
                        continue
                    seen_review_ids.add(row["review_id"])
                    collected_rows.append(row)
                    category_counts[category] += 1
                    if category_counts[category] >= per_category_target:
                        break

                page_positions[product.product_id] += 1

                if delay_seconds > 0:
                    time.sleep(delay_seconds)

                if category_counts[category] >= per_category_target:
                    break

            if not progressed:
                break

    metadata = {
        "category_counts": category_counts,
        "product_summaries": {
            str(product.product_id): {
                "category": product.category,
                "product_name": product.product_name,
                "comments_total": summaries[product.product_id].comments_total,
                "comments_good": summaries[product.product_id].comments_good,
                "default_good": summaries[product.product_id].default_good,
                "satisfy_rate": summaries[product.product_id].satisfy_rate,
            }
            for product in catalog
        },
        "requested_per_category_target": per_category_target,
        "collected_total": len(collected_rows),
    }
    return collected_rows, metadata


def group_products_by_category(
    products: Iterable[XiaomiProduct],
) -> dict[str, list[XiaomiProduct]]:
    grouped: dict[str, list[XiaomiProduct]] = defaultdict(list)
    for product in products:
        grouped[product.category].append(product)
    return dict(grouped)


def filter_catalog(categories: Iterable[str] | None = None) -> list[XiaomiProduct]:
    if categories is None:
        return list(XIAOMI_PRODUCT_CATALOG)
    normalized_categories = {normalize_whitespace(category) for category in categories}
    return [
        product
        for product in XIAOMI_PRODUCT_CATALOG
        if product.category in normalized_categories
    ]


def _request_jsonp(
    endpoint: str,
    params: dict[str, Any],
    referer: str,
    timeout: float,
) -> dict[str, Any]:
    request_params = dict(params)
    request_params["jsonpcallback"] = request_params.get("jsonpcallback", "codex_cb")
    url = f"{endpoint}?{urlencode(request_params)}"
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": referer,
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8", "ignore")
    parsed = parse_jsonp_payload(payload)
    if parsed.get("code") != 200:
        raise ValueError(f"Unexpected Xiaomi API response: {parsed.get('msg')}")
    return parsed


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
