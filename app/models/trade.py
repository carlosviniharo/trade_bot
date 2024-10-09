# models/user.py
from pydantic import BaseModel

class UserBase(BaseModel):
    name: str
    email: str
    age: int

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: str

class TradeData(BaseModel):
    symbol: str
    volume_change: float
    close: float
    r1: float
    s1: float
    r2: float
    s2: float
    r3: float
    s3: float

class CreateTrade(TradeData):
    pass

class Trade(TradeData):
    id: str