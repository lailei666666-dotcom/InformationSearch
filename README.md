# InformationSearch：商品评论反馈语义检索实验平台

InformationSearch 是一个面向中文商品评论的检索实验项目。它以小米商城商品评论为数据基础，围绕“用户需求理解”这个目标，构建并比较三类检索方法：

- 传统关键词检索：BM25
- 语义向量检索：本地语义编码或 OpenAI-compatible embedding
- 混合检索：BM25 与语义检索分数融合

项目适合用于课程实验、论文复现、信息检索学习、商品评论分析和用户需求理解研究。

如果你完全不了解这个项目，建议先阅读：[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)。

## 项目能解决什么问题

普通关键词搜索只能匹配表面的词，例如用户搜“静音键盘”，系统容易找到包含“静音”“键盘”的评论。

但真实用户需求经常是这样的：

```text
宿舍晚上写论文想要安静一点的键盘
```

这句话背后其实包含几个隐含需求：

- 使用场景：宿舍、晚上、写论文
- 核心需求：安静、不打扰别人
- 商品类别：键盘

本项目就是为了研究：如何从商品评论中找出真正能支持这些需求的商品和评论证据。

## 项目特点

- 支持完整实验流程：采集、清洗、建索引、检索、评测、可视化。
- 支持三种检索系统：BM25、语义检索、混合检索。
- 支持评论级和商品级两种结果。
- 支持将多条评论聚合成商品推荐结果，并保留评论证据。
- 自带小规模人工标注集，可直接跑 benchmark。
- 测试覆盖较完整，当前自动化测试为 `119 passed`。

## 技术栈

| 类型 | 技术 |
| --- | --- |
| 语言 | Python 3.11+ |
| 数据处理 | pandas、pyarrow |
| 配置读取 | PyYAML、pydantic |
| 关键词检索 | rank-bm25 |
| 中文检索处理 | 字符级切分、中文 bigram |
| 语义编码 | scikit-learn HashingVectorizer、TF-IDF、TruncatedSVD |
| 向量检索 | FAISS，缺失时可使用 NumPy fallback |
| 评测指标 | Precision@K、Recall@K、MRR |
| 可视化 | matplotlib |
| 测试 | pytest |

## 目录结构

```text
InformationSearch/
├── configs/                  # 实验配置，例如 embedding 模型、top-k、融合权重
├── data/
│   ├── raw/                  # 原始评论数据
│   ├── processed/            # 清洗后的评论数据和检索用数据
│   └── annotations/          # 查询集、评论相关性标注、商品相关性标注
├── docs/                     # 项目报告、设计文档、实施计划
├── outputs/                  # 本地生成的索引、实验结果、图表
├── scripts/
│   ├── collect/              # 数据采集脚本
│   ├── preprocess/           # 数据清洗和格式转换脚本
│   ├── build_index/          # BM25 和语义索引构建脚本
│   ├── run_retrieval/        # BM25、语义、混合检索入口
│   ├── evaluate/             # benchmark 和 smoke pipeline
│   └── visualize/            # 表格和图表生成
├── src/
│   ├── common/               # 公共工具、配置、聚合、查询解析
│   ├── traditional_retrieval/# BM25 传统检索
│   ├── semantic_retrieval/   # 语义向量检索
│   ├── hybrid_retrieval/     # 混合检索融合
│   └── evaluation/           # 评测指标和 qrels 加载
├── tests/                    # 自动化测试
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境准备

建议使用虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```bash
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果只使用本地语义编码，不需要任何外部 API。

如果使用在线 embedding，需要配置环境变量：

```bash
set OPENAI_API_KEY=你的 API Key
set OPENAI_BASE_URL=https://api.openai.com/v1
```

PowerShell 写法：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 数据说明

核心检索数据位于：

```text
data/processed/xiaomi_reviews_retrieval.csv
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `review_id` | 评论 ID |
| `product_id` | 商品 ID |
| `product_name` | 商品名称 |
| `category` | 商品类别 |
| `clean_text` | 清洗后的评论文本 |

人工评测数据位于：

```text
data/annotations/
```

包含：

- `queries.csv`：查询集
- `qrels_reviews.csv`：查询与相关评论的标注
- `qrels_products.csv`：查询与相关商品的标注

## 快速开始

下面命令默认在项目根目录执行。

### 1. 清洗评论数据

```bash
python scripts/preprocess/clean_reviews.py data/raw/xiaomi_reviews.csv data/processed/xiaomi_reviews_clean.csv
```

### 2. 生成检索用精简数据

```bash
python scripts/preprocess/slim_reviews_for_retrieval.py data/processed/xiaomi_reviews_clean.csv data/processed/xiaomi_reviews_retrieval.csv
```

### 3. 构建 BM25 索引

```bash
python scripts/build_index/build_bm25_index.py data/processed/xiaomi_reviews_retrieval.csv outputs/indexes/bm25_xiaomi.json
```

### 4. 构建本地语义索引

```bash
python scripts/build_index/build_local_semantic_index.py data/processed/xiaomi_reviews_retrieval.csv outputs/indexes/semantic_xiaomi.json outputs/indexes/semantic_xiaomi_encoder.pkl
```

### 5. 运行 BM25 检索

```bash
python scripts/run_retrieval/run_bm25.py outputs/indexes/bm25_xiaomi.json 静音键盘 --top-k 5
```

### 6. 运行语义检索

```bash
python scripts/run_retrieval/run_semantic.py outputs/indexes/semantic_xiaomi.json --query-text 宿舍晚上写论文想要安静一点的键盘 --encoder-path outputs/indexes/semantic_xiaomi_encoder.pkl --top-k 5
```

### 7. 运行混合检索

```bash
python scripts/run_retrieval/run_hybrid.py --query-text 宿舍晚上写论文想要安静一点的键盘 --bm25-index outputs/indexes/bm25_xiaomi.json --semantic-index outputs/indexes/semantic_xiaomi.json --encoder-path outputs/indexes/semantic_xiaomi_encoder.pkl --alpha 0.55 --top-k 5
```

### 8. 输出商品级聚合结果

加上 `--aggregate` 后，系统会把评论级结果聚合为商品级结果，并保留证据评论：

```bash
python scripts/run_retrieval/run_hybrid.py --query-text 宿舍晚上写论文想要安静一点的键盘 --bm25-index outputs/indexes/bm25_xiaomi.json --semantic-index outputs/indexes/semantic_xiaomi.json --encoder-path outputs/indexes/semantic_xiaomi_encoder.pkl --alpha 0.55 --aggregate --product-top-n 5 --evidence-top-n 3
```

## 一键 smoke 检查

如果只是想确认项目能跑通，可以执行：

```bash
python scripts/evaluate/run_smoke_pipeline.py data/processed/xiaomi_reviews_retrieval.csv --output outputs/smoke --query 静音键盘
```

该命令会：

1. 读取处理好的评论数据。
2. 构建一个 BM25 smoke 索引。
3. 执行一次查询。
4. 输出 smoke 结果和摘要。

## 完整 benchmark

运行三套系统的完整评测：

```bash
python scripts/evaluate/run_full_benchmark.py --annotations-dir data/annotations --bm25-index outputs/indexes/bm25_xiaomi.json --semantic-index outputs/indexes/semantic_xiaomi.json --encoder-path outputs/indexes/semantic_xiaomi_encoder.pkl --output-dir outputs/runs/xiaomi_benchmark --top-k 10 --product-top-n 10 --alpha 0.55
```

输出内容包括：

```text
outputs/runs/xiaomi_benchmark/
├── ranked/       # 每个查询的排序结果
├── benchmarks/   # 每个系统的评测指标
└── tables/       # 汇总表格
```

## 生成图表

根据 benchmark 汇总表生成柱状图：

```bash
python scripts/visualize/make_figures.py outputs/runs/xiaomi_benchmark/tables/benchmark_summary.csv --metric mrr --output outputs/runs/xiaomi_benchmark/figures/mrr.png
python scripts/visualize/make_figures.py outputs/runs/xiaomi_benchmark/tables/benchmark_summary.csv --metric precision_at_k --output outputs/runs/xiaomi_benchmark/figures/precision_at_k.png
python scripts/visualize/make_figures.py outputs/runs/xiaomi_benchmark/tables/benchmark_summary.csv --metric recall_at_k --output outputs/runs/xiaomi_benchmark/figures/recall_at_k.png
```

## 核心模块说明

### BM25 传统检索

核心代码：

```text
src/traditional_retrieval/bm25_engine.py
src/traditional_retrieval/tokenizer.py
```

BM25 适合处理关键词明确的查询。项目针对中文文本做了字符级和 bigram 处理，提高中文短文本匹配能力。

### 语义检索

核心代码：

```text
src/semantic_retrieval/local_encoder.py
src/semantic_retrieval/semantic_engine.py
src/semantic_retrieval/faiss_index.py
```

语义检索会把评论和查询变成向量，再计算向量相似度。它更适合处理口语化、场景化、隐含需求类查询。

### 混合检索

核心代码：

```text
src/hybrid_retrieval/fusion.py
```

混合检索会对 BM25 分数和语义检索分数做归一化，然后按权重融合：

```text
最终分数 = alpha * 语义分数 + (1 - alpha) * BM25 分数
```

### 商品聚合

核心代码：

```text
src/common/aggregation.py
```

评论级检索会返回一条条评论。商品聚合会把同一商品下的多条评论合并，得到商品级推荐结果，并保留若干条评论作为证据。

## 评测指标

| 指标 | 含义 |
| --- | --- |
| `precision_at_k` | 前 k 个结果中，有多少比例是相关结果 |
| `recall_at_k` | 所有相关结果中，有多少被前 k 个结果覆盖 |
| `mrr` | 第一个相关结果越靠前，分数越高 |

## 测试

运行全部测试：

```bash
python -m pytest -q
```

当前项目验证结果：

```text
119 passed
```

## GitHub 上传说明

本项目包含一些本地生成的大型索引和 embedding 缓存文件，例如：

```text
outputs/indexes/semantic_xiaomi_bailian.json
outputs/semantic/xiaomi_bailian_embeddings.jsonl
outputs/semantic/bailian_embedding_cache.jsonl
```

这些文件超过 GitHub 普通仓库的单文件大小限制，不建议直接提交。

仓库会保留源码、配置、数据、测试、文档和可复现实验脚本；大型索引文件可以通过 README 中的建索引命令在本地重新生成。

## 常见问题

### 1. 为什么不把所有 outputs 都上传？

`outputs/` 主要是生成产物，不是源码。部分索引文件超过 100MB，GitHub 普通仓库无法直接接收。项目保留了生成这些文件的脚本，所以可以本地复现。

### 2. 没有 API Key 能跑吗？

可以。使用本地语义编码器时不需要 API Key：

```bash
python scripts/build_index/build_local_semantic_index.py data/processed/xiaomi_reviews_retrieval.csv outputs/indexes/semantic_xiaomi.json outputs/indexes/semantic_xiaomi_encoder.pkl
```

### 3. 这个项目是 Web 应用吗？

不是。它是命令行实验平台，重点是检索算法、评测和实验复现。

### 4. 新手应该从哪里开始读代码？

建议按这个顺序：

1. `docs/PROJECT_REPORT.md`
2. `data/processed/xiaomi_reviews_retrieval.csv`
3. `src/traditional_retrieval/bm25_engine.py`
4. `src/semantic_retrieval/local_encoder.py`
5. `src/hybrid_retrieval/fusion.py`
6. `scripts/evaluate/run_full_benchmark.py`

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。
