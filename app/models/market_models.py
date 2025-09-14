# models/user.py
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional, List, TypeVar, Generic

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
    is_price_event: bool = False
    is_volume_event: bool = False
    price_rate: Optional[float] = 0
    volume_rate: Optional[float] = 0
    atr_pct: Optional[float] = 0
    close: float
    date_of_creation: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    date_of_modification: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    atr_results: List[AtrResult]


class Message(BaseModel):
    message: str
    success: bool


class MarketSentiment(BaseModel):
    report: str


class PaginatedResponse(BaseModel, Generic[T]):
    total: int
    page: int
    limit: int
    items: List[T]


class XGBoostPredictionResult(BaseModel):
    message: str
    time_frame: str
    prediction_confidence: float
    latest_predicted_resistance: float
    latest_predicted_support: float
