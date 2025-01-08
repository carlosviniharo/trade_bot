# services/user_service.py
from bson import ObjectId
from fastapi import HTTPException

from app.core.logging import AppLogger
from app.models.trade import User, UserCreate, StockChangeRecordCreate, StockChangeRecord, StockChangeRecordRead
from app.core.database import get_database
from app.utils.helper import BaseVolumeAnalyzer
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


async def list_stock_change_records():
    db = await get_database()
    stock_change_records = await db["stock_change_records"].find().to_list(1000)
    return [stock_change_record_helper(stock) for stock in stock_change_records]

async def get_atr(symbol: str):
    trade = BaseVolumeAnalyzer()
    try:
        await trade.initialize()
        await trade.get_historical_data(symbol)
        trade.calculate_atr()
        df = trade.get_df()
    except Exception as e:
        # Raising an HTTPException with a status code and the error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        await trade.close()
    logger.info(df.to_dict(orient='records')[-1])
    return df.to_dict(orient='records')[-1]

async def send_messages(message):
    whastapp = WhatsAppOutput(settings.WHATSAPP_TOKEN, settings.PHONE_NUMBER_ID)
    msg = message.model_dump()
    try:
        await whastapp.send_text_message("447729752680", msg["message"])
        return {"message": "Message sent successfully", "success": True}
    except Exception as e:
        # Raising an HTTPException with a status code and the error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    # finally:
    #     await whastapp.close()