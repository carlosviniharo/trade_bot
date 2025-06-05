# services/user_service.py
import asyncio

from bson import ObjectId
from fastapi import HTTPException

from app.core.logging import AppLogger
from app.models.trade import User, UserCreate, StockChangeRecordCreate, StockChangeRecord, StockChangeRecordRead, \
    MarketSentiment, ResistanceSupport, PaginatedResponse
from app.core.database import get_database
from app.utils.helper import BaseVolumeAnalyzer, format_symbol_name, MarketSentimentAnalyzer, PaginationParams
from app.utils.whatsapp_connector import WhatsAppOutput
from app.core.config import settings

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


def stock_change_record_helper(stock_change_record) -> StockChangeRecordRead:
    return StockChangeRecordRead(
        id=str(stock_change_record["_id"]),
        price_changes=stock_change_record["price_changes"],
        volume_changes=stock_change_record["volume_changes"],
    )


async def create_stock_change_records(data_stock_change_record: StockChangeRecordCreate):
    db = await get_database()
    new_stock_change_record = await db["stock_change_records"].insert_one(data_stock_change_record.model_dump())
    stock_change_record = await db["stock_change_records"].find_one({"_id": new_stock_change_record.inserted_id})
    return stock_change_record_helper(stock_change_record)


async def list_stock_change_records(params: PaginationParams):
    db = await get_database()
    # stock_change_records = await db["stock_change_records"].find().to_list(1000)
    # return [stock_change_record_helper(stock) for stock in stock_change_records]
    stock_change_records = db["stock_change_records"]

    total = await stock_change_records.count_documents({})
    cursor = (
        stock_change_records.find()
        .skip(params.skip)
        .limit(params.limit)
    )
    items = [stock_change_record_helper(stock) async for stock in cursor]

    return PaginatedResponse(
        total=total,
        page=params.page,
        limit=params.limit,
        items=items
    )

# TODO: Add the return model of atr instead of a dict
async def get_atr(symbol: str):
    trade = BaseVolumeAnalyzer()
    symbol = format_symbol_name(symbol)
    try:
        await trade.initialize()
        await trade.get_historical_data(symbol)
        trade.calculate_atr()
    except Exception as e:
        # Raising an HTTPException with a status code and the error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        await trade.close()
    return trade.get_df().to_dict(orient='records')[-1]


async def send_messages(message):
    whastapp = WhatsAppOutput(settings.WHATSAPP_TOKEN, settings.PHONE_NUMBER_ID)
    msg = message.model_dump()
    try:
        await whastapp.send_text_message("447729752680", msg["message"])
        return {"message": "Message sent successfully", "success": True}
    except Exception as e:
        # Raising an HTTPException with a status code and the error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


async def get_support_resistance_levels(symbol: str):
    trade = BaseVolumeAnalyzer()
    symbol =  format_symbol_name(symbol)

    try:
        await trade.initialize()
        await trade.get_historical_data(symbol)
        trade.calculate_support_resistance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        await trade.close()
    sup_res_resp = trade.get_df().iloc[-1]
    return ResistanceSupport(
        close=sup_res_resp["close"],
        pivot_point=sup_res_resp["pivot_point"],
        r1=sup_res_resp["r1"],
        s1=sup_res_resp["s1"],
        r2=sup_res_resp["r2"],
        s2=sup_res_resp["s2"],
        r3=sup_res_resp["r3"],
        s3=sup_res_resp["s3"]
    )


async def get_market_sentiment():
    analyzer = MarketSentimentAnalyzer()
    try:
        market_data = analyzer.fetch_market_data()
        sentiment_score = analyzer.calculate_weighted_sentiment(market_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    return MarketSentiment(
        report=analyzer.render_report(sentiment_score)
    )

