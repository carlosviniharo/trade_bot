# routers/user_router.py
from fastapi import APIRouter, HTTPException
from fastapi.params import Query, Depends
from app.models.market_models import (
    User,
    UserCreate,
    AtrData,
    MarketEvent,
    MarketEventCreate,
    Message,
    MarketSentiment,
    PaginatedResponse,
    XGBoostPredictionResult
)
from app.services.market_services import (
    create_user,
    get_user,
    list_users,
    update_user,
    delete_user,
    get_atr, 
    create_market_event,
    list_market_events,
    send_messages,
    get_support_resistance_levels,
    get_market_sentiment, send_messages_tg,
    get_xgboosr_prediction
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

@router.post("/marketEvents/", response_model=MarketEvent)
async def create_new_market_event(market_event: MarketEventCreate):
    return await create_market_event(market_event)

@router.get("/marketEvents/", response_model=PaginatedResponse)
async def get_list_market_events(params: PaginationParams = Depends()) -> PaginatedResponse:
    return await list_market_events(params)

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


@router.get("/marketSentiment/", response_model=MarketSentiment)
async def fetch_market_sentiment():
    sentiment = await get_market_sentiment()
    if sentiment is None:
        raise HTTPException(
            status_code=404,
            detail=f"The market sentiment calculation did not work")
    return sentiment

@router.get("/xgboostPrediction/", response_model=XGBoostPredictionResult)
async def get_xgboost_prediction(symbol: str = Query(...), time_frame: str = Query(...)):
    """
    Get XGBoost-based support and resistance prediction for a given symbol.
    """
    pred = await get_xgboosr_prediction(symbol, time_frame)
    if pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"The support and resistance calculation for'{symbol}' did not work")
    return pred
