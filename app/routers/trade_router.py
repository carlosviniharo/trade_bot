# routers/user_router.py
from fastapi import APIRouter, HTTPException
from fastapi.params import Query, Depends
from app.models.trade import (
    User,
    UserCreate,
    AtrData,
    StockChangeRecordRead,
    StockChangeRecordCreate,
    Message,
    ResistanceSupport,
    MarketSentiment,
    PaginatedResponse
)
from app.services.trade_service import (
    create_user,
    get_user,
    list_users,
    update_user,
    delete_user,
    get_atr, create_stock_change_records,
    list_stock_change_records,
    send_messages,
    get_support_resistance_levels,
    get_market_sentiment, send_messages_tg
)
from app.utils.helper import PaginationParams

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

@router.post("/stockChangeRecord/", response_model=StockChangeRecordRead)
async def create_new_stock_change_records(stock_change_record: StockChangeRecordCreate):
    return await create_stock_change_records(stock_change_record)

@router.get("/stockChangeRecord/", response_model=PaginatedResponse)
async def get_list_stock_change_records(params: PaginationParams = Depends()):
    return await list_stock_change_records(params)

@router.get("/getAtrBySymbol/", response_model=AtrData)
async def get_atr_by_symbol(symbol: str = Query(...)):
    atr = await get_atr(symbol)
    if atr is None:
        raise HTTPException(status_code=404, detail=f"ATR data for symbol '{symbol}' not found")
    return atr

@router.post("/sendMessageWP/", response_model=Message)
async def send_wp_message(message: Message):
    return await send_messages(message)

@router.post("/sendMessageTG/", response_model=Message)
async def send_tg_message(message: Message):
    return await send_messages_tg(message)


@router.get("/supportResistance/", response_model=ResistanceSupport)
async def get_support_resistance(symbol: str = Query(...)):
    sup_res = await get_support_resistance_levels(symbol)
    if sup_res is None:
        raise HTTPException(
            status_code=404,
            detail=f"The support and resistance calculation for'{symbol}' did not work")
    return sup_res

@router.get("/marketSentiment/", response_model=MarketSentiment)
async def fetch_market_sentiment():
    sentiment = await get_market_sentiment()
    if sentiment is None:
        raise HTTPException(
            status_code=404,
            detail=f"The market sentiment calculation did not work")
    return sentiment