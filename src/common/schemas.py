from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CategoriesSettings(StrictModel):
    categories: dict[str, list[str]]


class EmbeddingSettings(StrictModel):
    provider: str
    model: str
    batch_size: int = Field(gt=0)
    dimensions: int | None = Field(default=None, gt=0)


class ExperimentSettings(StrictModel):
    top_k: int = Field(gt=0)
    product_top_n: int = Field(gt=0)
    alpha: float = Field(ge=0.0, le=1.0)


class ReviewRecord(StrictModel):
    review_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    category: str = Field(pattern="^(键盘|台灯|耳机|笔记本|充电宝)$")
    raw_text: str = Field(min_length=1)
    clean_text: str = Field(min_length=1)


class Settings(StrictModel):
    categories: dict[str, list[str]]
    embedding: EmbeddingSettings
    experiment: ExperimentSettings
