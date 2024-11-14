# routers/user_router.py
from fastapi import APIRouter, HTTPException
from fastapi.params import Query

from app.core import logging
from app.models.trade import User, UserCreate, TradeData, TradeCreate, Trade, AtrData
from app.services.trade_service import (
    create_user,
    get_user,
    list_users,
    update_user,
    delete_user,
    create_trade,
    list_trades,
    get_atr
)

router = APIRouter()

@router.post("/users/", response_model=User)
async def create_new_user(user: UserCreate):
    return await create_user(user)

@router.get("/users/{user_id}", response_model=User)
async def get_user_by_id(user_id: str):
    user = await get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/users/", response_model=list[User])
async def get_all_users():
    return await list_users()

@router.put("/users/{user_id}", response_model=User)
async def update_user_by_id(user_id: str, user: UserCreate):
    updated_user = await update_user(user_id, user)
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user

@router.delete("/users/{user_id}")
async def delete_user_by_id(user_id: str):
    deleted_count = await delete_user(user_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted"}

@router.post("/trades/", response_model=Trade)
async def create_new_trade(trade: TradeCreate):
    return await create_trade(trade)

@router.get("/trades/", response_model=list[Trade])
async def get_all_trades():
    return await list_trades()

@router.get("/getAtrBySymbol/", response_model=AtrData)
async def get_atr_by_symbol(symbol: str = Query(...)):
    atr = await get_atr(symbol)
    if atr is None:
        raise HTTPException(status_code=404, detail=f"ATR data for symbol '{symbol}' not found")
    return atr[-1]