# services/user_service.py
from bson import ObjectId
from app.models.trade import User, UserCreate, Trade, TradeCreate
from app.core.database import get_database

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
    new_user = await db["users"].insert_one(user_data.dict())
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
    await db["users"].update_one({"_id": ObjectId(user_id)}, {"$set": user_data.dict()})
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    return user_helper(user) if user else None

async def delete_user(user_id: str):
    db = await get_database()
    result = await db["users"].delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count


def trade_helper(trade) -> Trade:
    return Trade(
        id=str(trade["_id"]),  # Assuming MongoDB ObjectId
        symbol=trade["symbol"],
        volume_change=trade["volume_change"],
        close=trade["close"],
        r1=trade["r1"],
        s1=trade["s1"],
        r2=trade["r2"],
        s2=trade["s2"],
        r3=trade["r3"],
        s3=trade["s3"],
        # date_of_creation=trade["date_of_creation"],
        # date_of_modification=trade["date_of_modification"]
    )

async def create_trade(trade_data: TradeCreate):
    db = await get_database()
    new_trade = await db["trades"].insert_one(trade_data.model_dump())
    trade = await db["trades"].find_one({"_id": new_trade.inserted_id})
    return trade_helper(trade)


async def list_trades():
    db = await get_database()
    trades = await db["trades"].find().to_list(2)
    return [trade_helper(trade) for trade in trades]