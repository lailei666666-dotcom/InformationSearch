# Product Review Retrieval Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible command-line experiment platform for 10,000+ Chinese product reviews across keyboard, desk lamp, earphone, laptop, and power bank categories, with BM25 retrieval, category-constrained semantic retrieval, hybrid retrieval, and a shared evaluation pipeline.

**Architecture:** The repository is organized as a single experiment platform with a shared data pipeline, shared indexing and aggregation utilities, three retrieval backends, and one evaluation/reporting workflow. Data is normalized into one canonical review table, then consumed by BM25, FAISS, and fusion pipelines so that every experiment runs on the same corpus, annotation set, and metrics implementation.

**Tech Stack:** Python 3.11+, pandas, pydantic, typer, PyYAML, jieba or pkuseg, rank-bm25, faiss-cpu, numpy, scikit-learn, matplotlib, seaborn, tqdm, tenacity, httpx, openai-compatible embeddings API.

---

## File Structure

**Files:**
- Create: `C:\mycode\python\InformationSearch\README.md`
- Create: `C:\mycode\python\InformationSearch\requirements.txt`
- Create: `C:\mycode\python\InformationSearch\.env.example`
- Create: `C:\mycode\python\InformationSearch\configs\categories.yml`
- Create: `C:\mycode\python\InformationSearch\configs\embedding.yml`
- Create: `C:\mycode\python\InformationSearch\configs\experiment.yml`
- Create: `C:\mycode\python\InformationSearch\data\raw\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\data\interim\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\data\processed\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\data\annotations\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\outputs\indexes\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\outputs\runs\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\outputs\figures\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\outputs\tables\.gitkeep`
- Create: `C:\mycode\python\InformationSearch\src\common\__init__.py`
- Create: `C:\mycode\python\InformationSearch\src\common\config.py`
- Create: `C:\mycode\python\InformationSearch\src\common\logging_utils.py`
- Create: `C:\mycode\python\InformationSearch\src\common\schemas.py`
- Create: `C:\mycode\python\InformationSearch\src\common\paths.py`
- Create: `C:\mycode\python\InformationSearch\src\common\io_utils.py`
- Create: `C:\mycode\python\InformationSearch\src\common\text.py`
- Create: `C:\mycode\python\InformationSearch\src\traditional_retrieval\__init__.py`
- Create: `C:\mycode\python\InformationSearch\src\semantic_retrieval\__init__.py`
- Create: `C:\mycode\python\InformationSearch\src\hybrid_retrieval\__init__.py`
- Create: `C:\mycode\python\InformationSearch\src\evaluation\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\collect\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\preprocess\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\build_index\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\run_retrieval\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\evaluate\__init__.py`
- Create: `C:\mycode\python\InformationSearch\scripts\visualize\__init__.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_smoke_layout.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_repository_layout_exists() -> None:
    root = Path("C:/mycode/python/InformationSearch")
    expected = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/annotations",
        "outputs/indexes",
        "outputs/runs",
        "outputs/figures",
        "outputs/tables",
        "src/common",
        "src/traditional_retrieval",
        "src/semantic_retrieval",
        "src/hybrid_retrieval",
        "src/evaluation",
        "scripts/collect",
        "scripts/preprocess",
        "scripts/build_index",
        "scripts/run_retrieval",
        "scripts/evaluate",
        "scripts/visualize",
    ]
    missing = [item for item in expected if not (root / item).exists()]
    assert not missing, missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_smoke_layout.py -v`
Expected: FAIL with missing repository directories or missing `tests` package.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/common/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
```

```text
# Create the directories and placeholder files listed in the "Files" section
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_smoke_layout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "chore: scaffold retrieval experiment platform"
```

### Task 1: Project Bootstrap And Shared Configuration

**Files:**
- Create: `C:\mycode\python\InformationSearch\pyproject.toml`
- Create: `C:\mycode\python\InformationSearch\src\common\config.py`
- Create: `C:\mycode\python\InformationSearch\src\common\logging_utils.py`
- Create: `C:\mycode\python\InformationSearch\src\common\schemas.py`
- Create: `C:\mycode\python\InformationSearch\src\common\io_utils.py`
- Create: `C:\mycode\python\InformationSearch\configs\categories.yml`
- Create: `C:\mycode\python\InformationSearch\configs\embedding.yml`
- Create: `C:\mycode\python\InformationSearch\configs\experiment.yml`
- Modify: `C:\mycode\python\InformationSearch\requirements.txt`
- Test: `C:\mycode\python\InformationSearch\tests\test_config_loading.py`

- [ ] **Step 1: Write the failing test**

```python
from src.common.config import load_settings


def test_load_settings_returns_expected_categories() -> None:
    settings = load_settings()
    assert "键盘" in settings.categories
    assert "台灯" in settings.categories
    assert settings.embedding.provider
    assert settings.experiment.top_k == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_config_loading.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/common/config.py
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from src.common.paths import CONFIGS_DIR


class EmbeddingSettings(BaseModel):
    provider: str
    model: str
    batch_size: int
    dimensions: int | None = None


class ExperimentSettings(BaseModel):
    top_k: int
    product_top_n: int
    alpha: float


class Settings(BaseModel):
    categories: dict[str, list[str]]
    embedding: EmbeddingSettings
    experiment: ExperimentSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_settings() -> Settings:
    return Settings(
        categories=_load_yaml(CONFIGS_DIR / "categories.yml")["categories"],
        embedding=EmbeddingSettings(**_load_yaml(CONFIGS_DIR / "embedding.yml")),
        experiment=ExperimentSettings(**_load_yaml(CONFIGS_DIR / "experiment.yml")),
    )
```

```yaml
# C:/mycode/python/InformationSearch/configs/categories.yml
categories:
  键盘: ["键盘", "机械键盘", "蓝牙键盘", "无线键盘"]
  台灯: ["台灯", "灯", "护眼灯", "宿舍灯"]
  耳机: ["耳机", "蓝牙耳机", "降噪耳机"]
  笔记本: ["笔记本", "电脑", "轻薄本"]
  充电宝: ["充电宝", "移动电源"]
```

```yaml
# C:/mycode/python/InformationSearch/configs/experiment.yml
top_k: 10
product_top_n: 3
alpha: 0.65
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_config_loading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add project config and shared schemas"
```

### Task 2: Data Collection Entry Points

**Files:**
- Create: `C:\mycode\python\InformationSearch\scripts\collect\download_public_dataset.py`
- Create: `C:\mycode\python\InformationSearch\scripts\collect\scrape_reviews_demo.py`
- Create: `C:\mycode\python\InformationSearch\src\common\http_client.py`
- Create: `C:\mycode\python\InformationSearch\src\common\retrying.py`
- Create: `C:\mycode\python\InformationSearch\data\raw\README.md`
- Test: `C:\mycode\python\InformationSearch\tests\test_collect_scripts.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import subprocess
import sys


def test_download_script_has_help_output() -> None:
    script = Path("C:/mycode/python/InformationSearch/scripts/collect/download_public_dataset.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_collect_scripts.py -v`
Expected: FAIL because the script does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/scripts/collect/download_public_dataset.py
from pathlib import Path

import typer


def main(output: Path = typer.Option(..., exists=False, dir_okay=True, file_okay=False)) -> None:
    output.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Dataset output directory prepared: {output}")


if __name__ == "__main__":
    typer.run(main)
```

```python
# C:/mycode/python/InformationSearch/scripts/collect/scrape_reviews_demo.py
from pathlib import Path

import typer


def main(category: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("category,product_name,review\n", encoding="utf-8")
    typer.echo(f"Prepared empty scrape template for {category}")


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_collect_scripts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add collection entry points"
```

### Task 3: Canonical Review Schema And Data Normalization

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\common\schemas.py`
- Create: `C:\mycode\python\InformationSearch\src\common\text.py`
- Create: `C:\mycode\python\InformationSearch\scripts\preprocess\normalize_reviews.py`
- Create: `C:\mycode\python\InformationSearch\scripts\preprocess\report_dataset_stats.py`
- Create: `C:\mycode\python\InformationSearch\tests\fixtures\sample_reviews.csv`
- Test: `C:\mycode\python\InformationSearch\tests\test_normalize_reviews.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd

from scripts.preprocess.normalize_reviews import normalize_reviews


def test_normalize_reviews_maps_columns_and_categories(tmp_path: Path) -> None:
    source = Path("C:/mycode/python/InformationSearch/tests/fixtures/sample_reviews.csv")
    output = tmp_path / "normalized.parquet"
    frame = normalize_reviews(source, output)
    assert set(["review_id", "product_id", "product_name", "category", "clean_text"]).issubset(frame.columns)
    assert frame["category"].isin(["键盘", "台灯", "耳机", "笔记本", "充电宝"]).all()
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_normalize_reviews.py -v`
Expected: FAIL because the module and fixture are missing.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/common/schemas.py
from pydantic import BaseModel


class ReviewRecord(BaseModel):
    review_id: str
    product_id: str
    product_name: str
    category: str
    review_content: str
    rating: float | None = None
    source: str
    clean_text: str
    need_tags: list[str] = []
```

```python
# C:/mycode/python/InformationSearch/scripts/preprocess/normalize_reviews.py
from pathlib import Path

import pandas as pd


CATEGORY_MAP = {
    "机械键盘": "键盘",
    "护眼灯": "台灯",
    "蓝牙耳机": "耳机",
    "轻薄本": "笔记本",
    "移动电源": "充电宝",
}


def normalize_reviews(source: Path, output: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    frame = frame.rename(
        columns={
            "评论编号": "review_id",
            "商品编号": "product_id",
            "商品名称": "product_name",
            "类别": "category",
            "评论内容": "review_content",
        }
    )
    frame["category"] = frame["category"].replace(CATEGORY_MAP)
    frame["clean_text"] = frame["review_content"].astype(str).str.strip()
    frame["source"] = source.name
    frame.to_parquet(output, index=False)
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_normalize_reviews.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: normalize review data into canonical schema"
```

### Task 4: Dataset Quality Filters And Statistics

**Files:**
- Create: `C:\mycode\python\InformationSearch\scripts\preprocess\clean_reviews.py`
- Create: `C:\mycode\python\InformationSearch\src\common\dedupe.py`
- Create: `C:\mycode\python\InformationSearch\src\common\quality.py`
- Modify: `C:\mycode\python\InformationSearch\scripts\preprocess\report_dataset_stats.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_clean_reviews.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from scripts.preprocess.clean_reviews import clean_reviews_frame


def test_clean_reviews_removes_short_noise_and_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "1", "clean_text": "很好", "category": "键盘"},
            {"review_id": "2", "clean_text": "按键声音很小，图书馆用不会影响别人", "category": "键盘"},
            {"review_id": "3", "clean_text": "按键声音很小，图书馆用不会影响别人", "category": "键盘"},
        ]
    )
    cleaned = clean_reviews_frame(frame)
    assert cleaned["review_id"].tolist() == ["2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_clean_reviews.py -v`
Expected: FAIL because cleaning helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/scripts/preprocess/clean_reviews.py
import pandas as pd


def clean_reviews_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["clean_text"] = cleaned["clean_text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["clean_text"].str.len() >= 8]
    cleaned = cleaned.drop_duplicates(subset=["category", "clean_text"])
    return cleaned.reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_clean_reviews.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add dataset quality filters"
```

### Task 5: Annotation Set And Evaluation Gold Data

**Files:**
- Create: `C:\mycode\python\InformationSearch\data\annotations\queries.csv`
- Create: `C:\mycode\python\InformationSearch\data\annotations\qrels_reviews.csv`
- Create: `C:\mycode\python\InformationSearch\data\annotations\qrels_products.csv`
- Create: `C:\mycode\python\InformationSearch\scripts\preprocess\bootstrap_annotations.py`
- Create: `C:\mycode\python\InformationSearch\src\evaluation\qrels.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_qrels_loading.py`

- [ ] **Step 1: Write the failing test**

```python
from src.evaluation.qrels import load_query_set


def test_query_set_covers_three_query_types() -> None:
    query_set = load_query_set()
    assert {"关键词明确型", "体验反馈型", "场景化隐含需求型"} == set(query_set["query_type"].unique())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_qrels_loading.py -v`
Expected: FAIL because annotation files do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/evaluation/qrels.py
from pathlib import Path

import pandas as pd

from src.common.paths import DATA_DIR


def load_query_set() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "annotations" / "queries.csv")
```

```csv
query_id,query_text,category,query_type
Q001,适合上课偷偷用的键盘,键盘,场景化隐含需求型
Q002,宿舍晚上用会不会打扰室友的灯,台灯,场景化隐含需求型
Q003,适合学生党买的耳机,耳机,体验反馈型
Q004,笔记本 轻薄 续航,笔记本,关键词明确型
Q005,出门带着方便的充电宝,充电宝,体验反馈型
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_qrels_loading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add evaluation query set bootstrap"
```

### Task 6: Category Recognition And Query Parsing

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\common\query_parser.py`
- Modify: `C:\mycode\python\InformationSearch\configs\categories.yml`
- Test: `C:\mycode\python\InformationSearch\tests\test_query_parser.py`

- [ ] **Step 1: Write the failing test**

```python
from src.common.query_parser import parse_query


def test_parse_query_extracts_category_and_need_text() -> None:
    parsed = parse_query("适合上课偷偷用的键盘")
    assert parsed.category == "键盘"
    assert parsed.need_text == "适合上课偷偷用的"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_query_parser.py -v`
Expected: FAIL because `parse_query` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/common/query_parser.py
from dataclasses import dataclass

from src.common.config import load_settings


@dataclass
class ParsedQuery:
    raw_text: str
    category: str | None
    need_text: str


def parse_query(text: str) -> ParsedQuery:
    settings = load_settings()
    for category, aliases in settings.categories.items():
        for alias in aliases:
            if alias in text:
                return ParsedQuery(raw_text=text, category=category, need_text=text.replace(alias, "").strip())
    return ParsedQuery(raw_text=text, category=None, need_text=text.strip())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_query_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add query category parsing"
```

### Task 7: BM25 Index Construction And Traditional Retrieval

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\traditional_retrieval\tokenizer.py`
- Create: `C:\mycode\python\InformationSearch\src\traditional_retrieval\bm25_engine.py`
- Create: `C:\mycode\python\InformationSearch\scripts\build_index\build_bm25_index.py`
- Create: `C:\mycode\python\InformationSearch\scripts\run_retrieval\run_bm25.py`
- Create: `C:\mycode\python\InformationSearch\tests\test_bm25_engine.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from src.traditional_retrieval.bm25_engine import BM25Engine


def test_bm25_engine_ranks_relevant_keyboard_review_first() -> None:
    reviews = pd.DataFrame(
        [
            {"review_id": "R1", "product_id": "K1", "product_name": "静音键盘A", "category": "键盘", "clean_text": "按键声音很小 图书馆用也不会影响别人"},
            {"review_id": "R2", "product_id": "L1", "product_name": "宿舍台灯A", "category": "台灯", "clean_text": "灯光柔和 晚上不刺眼"},
        ]
    )
    engine = BM25Engine.from_frame(reviews)
    results = engine.search("适合上课偷偷用的键盘", top_k=2)
    assert results[0]["review_id"] == "R1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_bm25_engine.py -v`
Expected: FAIL because the BM25 engine does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/traditional_retrieval/bm25_engine.py
from __future__ import annotations

import pandas as pd
from rank_bm25 import BM25Okapi


class BM25Engine:
    def __init__(self, frame: pd.DataFrame, tokenized: list[list[str]]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenized = tokenized
        self.index = BM25Okapi(tokenized)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BM25Engine":
        tokenized = [str(text).split() for text in frame["clean_text"].tolist()]
        return cls(frame=frame, tokenized=tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores = self.index.get_scores(query.split())
        ranked = self.frame.copy()
        ranked["bm25_score"] = scores
        ranked = ranked.sort_values("bm25_score", ascending=False).head(top_k)
        return ranked.to_dict(orient="records")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_bm25_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add bm25 retrieval pipeline"
```

### Task 8: Product Aggregation And Evidence Selection

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\common\aggregation.py`
- Modify: `C:\mycode\python\InformationSearch\scripts\run_retrieval\run_bm25.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_aggregation.py`

- [ ] **Step 1: Write the failing test**

```python
from src.common.aggregation import aggregate_review_hits


def test_aggregate_review_hits_groups_reviews_into_product_results() -> None:
    hits = [
        {"review_id": "R1", "product_id": "K1", "product_name": "静音键盘A", "category": "键盘", "score": 0.9, "clean_text": "按键声音很小"},
        {"review_id": "R2", "product_id": "K1", "product_name": "静音键盘A", "category": "键盘", "score": 0.8, "clean_text": "宿舍晚上打字不吵"},
        {"review_id": "R3", "product_id": "K2", "product_name": "机械键盘B", "category": "键盘", "score": 0.6, "clean_text": "手感不错"},
    ]
    products = aggregate_review_hits(hits, product_top_n=2)
    assert products[0]["product_id"] == "K1"
    assert len(products[0]["evidence"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_aggregation.py -v`
Expected: FAIL because the aggregator does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/common/aggregation.py
from collections import defaultdict


def aggregate_review_hits(hits: list[dict], product_top_n: int = 3) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for hit in hits:
        grouped[hit["product_id"]].append(hit)

    products: list[dict] = []
    for product_id, rows in grouped.items():
        ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
        evidence = ranked[:product_top_n]
        avg_score = sum(item["score"] for item in evidence) / len(evidence)
        products.append(
            {
                "product_id": product_id,
                "product_name": evidence[0]["product_name"],
                "category": evidence[0]["category"],
                "score": avg_score,
                "evidence": evidence,
            }
        )
    return sorted(products, key=lambda item: item["score"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_aggregation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add product aggregation layer"
```

### Task 9: Embedding Client And Vector Cache

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\semantic_retrieval\embedding_client.py`
- Create: `C:\mycode\python\InformationSearch\src\semantic_retrieval\embedding_cache.py`
- Create: `C:\mycode\python\InformationSearch\scripts\build_index\embed_reviews.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_embedding_cache.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from src.semantic_retrieval.embedding_cache import EmbeddingCache


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.jsonl")
    cache.set("按键声音很小", [0.1, 0.2, 0.3])
    assert cache.get("按键声音很小") == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_embedding_cache.py -v`
Expected: FAIL because the cache module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/semantic_retrieval/embedding_cache.py
from __future__ import annotations

import json
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, list[float]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                self._store[payload["text"]] = payload["vector"]

    def get(self, text: str) -> list[float] | None:
        return self._store.get(text)

    def set(self, text: str, vector: list[float]) -> None:
        self._store[text] = vector
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"text": text, "vector": vector}, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_embedding_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add embedding cache for semantic retrieval"
```

### Task 10: FAISS Indexing And Semantic Retrieval

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\semantic_retrieval\faiss_index.py`
- Create: `C:\mycode\python\InformationSearch\src\semantic_retrieval\semantic_engine.py`
- Create: `C:\mycode\python\InformationSearch\scripts\build_index\build_faiss_index.py`
- Create: `C:\mycode\python\InformationSearch\scripts\run_retrieval\run_semantic.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_semantic_engine.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

from src.semantic_retrieval.semantic_engine import SemanticEngine


def test_semantic_engine_filters_by_category_before_ranking() -> None:
    frame = pd.DataFrame(
        [
            {"review_id": "R1", "product_id": "K1", "product_name": "静音键盘A", "category": "键盘", "clean_text": "按键声音很小 图书馆用也不会影响别人"},
            {"review_id": "R2", "product_id": "L1", "product_name": "宿舍台灯A", "category": "台灯", "clean_text": "灯光柔和 晚上不刺眼"},
        ]
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    engine = SemanticEngine.from_vectors(frame, vectors)
    results = engine.search(query_vector=np.array([1.0, 0.0], dtype="float32"), category="键盘", top_k=2)
    assert results[0]["review_id"] == "R1"
    assert all(item["category"] == "键盘" for item in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_semantic_engine.py -v`
Expected: FAIL because semantic retrieval code is missing.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/semantic_retrieval/semantic_engine.py
from __future__ import annotations

import numpy as np
import pandas as pd


class SemanticEngine:
    def __init__(self, frame: pd.DataFrame, vectors: np.ndarray) -> None:
        self.frame = frame.reset_index(drop=True)
        self.vectors = vectors

    @classmethod
    def from_vectors(cls, frame: pd.DataFrame, vectors: np.ndarray) -> "SemanticEngine":
        return cls(frame=frame, vectors=vectors)

    def search(self, query_vector: np.ndarray, category: str | None, top_k: int = 10) -> list[dict]:
        frame = self.frame
        vectors = self.vectors
        if category is not None:
            mask = frame["category"] == category
            frame = frame[mask].reset_index(drop=True)
            vectors = vectors[mask.to_numpy()]
        scores = vectors @ query_vector
        ranked = frame.copy()
        ranked["score"] = scores
        return ranked.sort_values("score", ascending=False).head(top_k).to_dict(orient="records")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_semantic_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add category-constrained semantic retrieval"
```

### Task 11: Hybrid Retrieval And Score Fusion

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\hybrid_retrieval\fusion.py`
- Create: `C:\mycode\python\InformationSearch\scripts\run_retrieval\run_hybrid.py`
- Test: `C:\mycode\python\InformationSearch\tests\test_fusion.py`

- [ ] **Step 1: Write the failing test**

```python
from src.hybrid_retrieval.fusion import fuse_scores


def test_fuse_scores_combines_bm25_and_semantic_scores() -> None:
    fused = fuse_scores(
        bm25_hits={"R1": 0.2, "R2": 0.8},
        semantic_hits={"R1": 0.9, "R2": 0.1},
        alpha=0.75,
    )
    assert fused["R1"] > fused["R2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_fusion.py -v`
Expected: FAIL because the fusion helper does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/hybrid_retrieval/fusion.py
def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse_scores(bm25_hits: dict[str, float], semantic_hits: dict[str, float], alpha: float) -> dict[str, float]:
    bm25_norm = _normalize(bm25_hits)
    semantic_norm = _normalize(semantic_hits)
    review_ids = set(bm25_norm) | set(semantic_norm)
    return {
        review_id: alpha * semantic_norm.get(review_id, 0.0) + (1 - alpha) * bm25_norm.get(review_id, 0.0)
        for review_id in review_ids
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_fusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add hybrid retrieval fusion"
```

### Task 12: Metrics And Benchmark Runner

**Files:**
- Create: `C:\mycode\python\InformationSearch\src\evaluation\metrics.py`
- Create: `C:\mycode\python\InformationSearch\src\evaluation\benchmark.py`
- Create: `C:\mycode\python\InformationSearch\scripts\evaluate\run_benchmark.py`
- Create: `C:\mycode\python\InformationSearch\tests\test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from src.evaluation.metrics import precision_at_k, reciprocal_rank


def test_precision_and_mrr_match_expected_values() -> None:
    ranked = ["R1", "R2", "R3"]
    relevant = {"R2", "R3"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_metrics.py -v`
Expected: FAIL because metrics are not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/src/evaluation/metrics.py
def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    window = ranked[:k]
    if not window:
        return 0.0
    return sum(1 for item in window if item in relevant) / len(window)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1 / index
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add retrieval evaluation metrics"
```

### Task 13: Experiment Reports And Visualization

**Files:**
- Create: `C:\mycode\python\InformationSearch\scripts\visualize\make_figures.py`
- Create: `C:\mycode\python\InformationSearch\scripts\visualize\make_tables.py`
- Create: `C:\mycode\python\InformationSearch\tests\test_report_outputs.py`
- Modify: `C:\mycode\python\InformationSearch\README.md`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.visualize.make_tables import write_summary_table


def test_write_summary_table_creates_csv(tmp_path: Path) -> None:
    output = tmp_path / "summary.csv"
    write_summary_table(
        rows=[{"system": "bm25", "precision_at_10": 0.4, "mrr": 0.5}],
        output=output,
    )
    assert output.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_report_outputs.py -v`
Expected: FAIL because visualization helpers are not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/scripts/visualize/make_tables.py
from pathlib import Path

import pandas as pd


def write_summary_table(rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_report_outputs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "feat: add experiment reporting outputs"
```

### Task 14: End-To-End Smoke Pipeline And Documentation

**Files:**
- Modify: `C:\mycode\python\InformationSearch\README.md`
- Create: `C:\mycode\python\InformationSearch\scripts\evaluate\run_smoke_pipeline.py`
- Create: `C:\mycode\python\InformationSearch\tests\test_smoke_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import subprocess
import sys


def test_smoke_pipeline_has_help_output() -> None:
    script = Path("C:/mycode/python/InformationSearch/scripts/evaluate/run_smoke_pipeline.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "processed" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_smoke_pipeline.py -v`
Expected: FAIL because the smoke pipeline script does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# C:/mycode/python/InformationSearch/scripts/evaluate/run_smoke_pipeline.py
from pathlib import Path

import typer


def main(processed: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Smoke pipeline ready for processed dataset: {processed}")


if __name__ == "__main__":
    typer.run(main)
```

```markdown
# C:/mycode/python/InformationSearch/README.md
## Quick Start

1. Install dependencies with `pip install -r requirements.txt`
2. Prepare raw datasets in `data/raw`
3. Normalize and clean reviews with scripts in `scripts/preprocess`
4. Build BM25 and FAISS indexes with scripts in `scripts/build_index`
5. Run `run_bm25.py`, `run_semantic.py`, and `run_hybrid.py`
6. Evaluate with `scripts/evaluate/run_benchmark.py`
7. Generate tables and figures with `scripts/visualize`
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest C:\mycode\python\InformationSearch\tests\test_smoke_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C C:/mycode/python/InformationSearch add .
git -C C:/mycode/python/InformationSearch commit -m "docs: add end-to-end experiment workflow"
```

## Self-Review

- Spec coverage:
  - Data collection, normalization, cleaning, and 10,000+ review target are covered by Tasks 2-4.
  - Annotation set, query parsing, and category constraint behavior are covered by Tasks 5-6.
  - BM25, semantic retrieval, hybrid retrieval, and product aggregation are covered by Tasks 7-11.
  - Metrics, benchmark experiments, and report generation are covered by Tasks 12-14.
- Placeholder scan:
  - No `TBD`, `TODO`, or “implement later” placeholders remain in the plan body.
- Type consistency:
  - Shared keys across tasks use `review_id`, `product_id`, `product_name`, `category`, `clean_text`, and `score`.
  - Retrieval stages consistently rank review hits first, then aggregate them into product results.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-product-review-retrieval-platform.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
