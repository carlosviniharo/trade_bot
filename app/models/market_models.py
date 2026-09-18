# models/user.py
from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class UserBase(BaseModel):
    name: str
    email: str
    age: int


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: str


class MarketEvent(BaseModel):
    symbol: str
    event_timestamp: datetime
    price_rate: float | None = 0
    atr_pct: float | None = 0
    close: float
    date_of_creation: datetime | None = Field(default_factory=lambda: datetime.now(UTC))
    date_of_modification: datetime | None = Field(default_factory=lambda: datetime.now(UTC))


class MarketEventCreate(MarketEvent):
    pass


class MarketEventRead(MarketEvent):
    id: str


class AtrResult(BaseModel):
    timeframe: str
    atr: float
    atr_pct: float
    atr_above_mean: bool


class AtrResults(BaseModel):
    timestamp: datetime | None = Field(default_factory=lambda: datetime.now(UTC))
    atr_results: list[AtrResult]


class Message(BaseModel):
    message: str
    success: bool


class MarketSentiment(BaseModel):
    report: str


class PaginatedResponse(BaseModel, Generic[T]):
    total: int
    page: int
    limit: int
    items: list[T]


class XGBoostPredictionResult(BaseModel):
    time_frame: str
    current_price: float
    resistance: float
    support: float
    upside_pct: float
    downside_pct: float
    risk_reward_ratio: float
    timestamp: datetime
    prediction_time_ms: float


class MarketTrendLabel(BaseModel):
    close: float
    trend: int
    timestamp: datetime | None = Field(default_factory=lambda: datetime.now(UTC))
