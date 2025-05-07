# models/user.py
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional, List


class UserBase(BaseModel):
    name: str
    email: str
    age: int

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: str


class SymbolChange(BaseModel):
    symbol: str
    price_change: Optional[float] = None
    volume_change: Optional[float] = None
    atr_pct: Optional[float] = None
    close: float


# Define the StockChangeRecord model
class StockChangeRecord(BaseModel):
    price_changes: Optional[List[SymbolChange]] = []
    volume_changes: Optional[List[SymbolChange]] = []
    date_of_creation: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    date_of_modification: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_modification_date(self):
        self.date_of_modification = datetime.now(timezone.utc)

class StockChangeRecordCreate(StockChangeRecord):
    pass

class StockChangeRecordRead(StockChangeRecord):
    id: str


class AtrData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float
    atr_pct: float

class Message(BaseModel):
    message: str
    success: bool

class ResistanceSupport(BaseModel):
    close: float
    pivot_point: float
    r1: float
    s1: float
    r2: float
    s2: float
    r3: float
    s3: float