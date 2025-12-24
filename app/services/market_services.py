# services/user_service.py
import asyncio
from typing import List

from bson import ObjectId
from fastapi import HTTPException
import pandas as pd

from app.core.logging import AppLogger
from app.models.market_models import (
    AtrResult,
    AtrResults,
    User,
    UserCreate, 
    MarketEvent, 
    MarketEventCreate, 
    MarketEventRead, 
    MarketSentiment,
    PaginatedResponse,
    XGBoostPredictionResult
    )
from app.core.database import get_database
from app.utils.helper import (
    BaseAnalyzer,
    BinanceVolumeAnalyzer,
    IndicatorComputer, 
    MarketSentimentAnalyzer, 
    PaginationParams,
    XGBoostSupportResistancePredictor, 
    format_symbol_name
    )
from app.utils.whatsapp_connector import WhatsAppOutput
from app.core.config import settings
from app.utils.telegram_connector import TelegramOutput

# Initialize logging
logger = AppLogger.get_logger()

# Helper to convert BSON ObjectId to string and format the user data
def user_helper(user) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "age": user["age"]
    }


async def create_user(user_data: UserCreate):
    db = await get_database()
    new_user = await db["users"].insert_one(user_data.model_dump())
    user = await db["users"].find_one({"_id": new_user.inserted_id})
    return user_helper(user)


async def get_user(user_id: str):
    db = await get_database()
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if user:
        return user_helper(user)
    return None


async def list_users():
    db = await get_database()
    users = await db["users"].find().to_list(1000)
    return [user_helper(user) for user in users]


async def update_user(user_id: str, user_data: UserCreate):
    db = await get_database()
    await db["users"].update_one({"_id": ObjectId(user_id)}, {"$set": user_data.model_dump()})
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    return user_helper(user) if user else None


async def delete_user(user_id: str):
    db = await get_database()
    result = await db["users"].delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count


def market_events_helper(market_event_record) -> MarketEventRead:
    return MarketEventRead(
        id=str(market_event_record["_id"]),
        symbol=market_event_record["symbol"],
        event_timestamp=market_event_record["event_timestamp"],
        is_price_event=market_event_record["is_price_event"],
        is_volume_event=market_event_record["is_volume_event"],
        price_rate=market_event_record["price_rate"],
        volume_rate=market_event_record["volume_rate"],
        atr_pct=market_event_record["atr_pct"],
        close=market_event_record["close"],
        date_of_creation=market_event_record["date_of_creation"],
        date_of_modification=market_event_record["date_of_modification"]
    )


async def create_market_event(market_event: MarketEventCreate):
    db = await get_database()
    new_market_event = await db["market_events"].insert_one(market_event.model_dump())
    record = await db["market_events"].find_one({"_id": new_market_event.inserted_id})
    return market_events_helper(record)


async def get_online_market_event() -> List[MarketEvent]:
    analyzer = BinanceVolumeAnalyzer()

    try:
        await analyzer.initialize()
        await analyzer.calculate_market_spikes()
        
        df_top_price_increase = analyzer.get_top_symbols(metric="price_rate")
        df_top_price_decrease = analyzer.get_top_symbols(metric="price_rate", ascending=True)

        df_merged = pd.concat([df_top_price_increase, df_top_price_decrease])
        return [MarketEvent(**event) for event in df_merged.to_dict(orient="records")]
   
    finally:
        await analyzer.close()


async def list_market_events(params: PaginationParams):
    db = await get_database()
    market_events_collection = db["market_events"]

    total = await market_events_collection.count_documents({})
    cursor = (
        market_events_collection.find()
        .skip(params.skip)
        .limit(params.limit)
    )
    items = [market_events_helper(event) async for event in cursor]

    return PaginatedResponse(
        total=total,
        page=params.page,
        limit=params.limit,
        items=items
    )


async def compute_atr_from_df(df: pd.DataFrame, timeframe: str):
    indicator = IndicatorComputer(df.copy())
    await indicator.run_in_thread(lambda: indicator.compute_atr().compute_atr_above_mean())
    df_atr = indicator.get_df_transformed()

    if df_atr.empty:
        return None

    payload = df_atr.iloc[-1].to_dict()
    payload["timeframe"] = timeframe
    return AtrResult(**payload)


async def get_atr(symbol: str) -> AtrResults:
    symbol = format_symbol_name(symbol)
    analyzer = BaseAnalyzer()

    try:
        await analyzer.initialize()

        async def fetch(tf: str):
            df = await analyzer.get_historical_data(symbol, tf, limit=500)
            return await compute_atr_from_df(df, tf)

        timeframes = ["1m", "5m", "15m"]

        results = await asyncio.gather(
            *(fetch(tf) for tf in timeframes),
            return_exceptions=True
        )

        return AtrResults(atr_results=results)

    finally:
        await analyzer.close()


async def send_messages(message):
    whatsapp = WhatsAppOutput(settings.WHATSAPP_TOKEN, settings.PHONE_NUMBER_ID)
    msg = message.model_dump()
    try:
        await whatsapp.send_text_message("447729752680", msg["message"])
    except Exception as e:
        # Raising an HTTPException with a status code and the error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    return {"message": "Message sent successfully", "success": True}


async def send_messages_tg(message):
    telegram = TelegramOutput(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)
    msg = message.model_dump()
    try:
        await telegram.send_text_message(msg["message"])
    except Exception as e:
        logger.exception(f"Failed to send message to Telegram: {e}")
        raise HTTPException(status_code=502, detail=f"Telegram delivery failed: {str(e)}")
    finally:
        await telegram.close()

    return {"success": True, "message": "Message successfully delivered to Telegram"}


async def get_market_sentiment() -> MarketSentiment:
    analyzer = MarketSentimentAnalyzer()
    try:
        market_data = analyzer.fetch_market_data()
        sentiment_score = analyzer.calculate_weighted_sentiment(market_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    return MarketSentiment(
        report=analyzer.render_report(sentiment_score)
        )


async def get_xgboosr_prediction(symbol: str, time_frame: str) -> XGBoostPredictionResult:
    predictor = XGBoostSupportResistancePredictor(
        window=10,
        n_splits=5,
        tune_hyperparams=True,  # Set to False for faster training
        use_optuna=True  # Set to False to use GridSearchCV instead
    )
    symbol = format_symbol_name(symbol)

    try:
        await predictor.initialize()
        df = await predictor.get_historical_data(symbol, timeframe=time_frame, limit=1000)
        
        logger.info(f"\nData shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info("\n" + "=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
    
        await predictor.train(df)

        logger.info("\n" + "=" * 60)

        prediction =await predictor.predict_latest()
        prediction['time_frame'] = time_frame
        return XGBoostPredictionResult(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        await predictor.close()