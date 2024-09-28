from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseSettings


# Load settings from environment variables
class Settings(BaseSettings):
    mongodb_uri: str
    mongodb_name: str

    class Config:
        env_file = ".env"


settings = Settings()


class Database:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        cls.client = AsyncIOMotorClient(settings.mongodb_uri)
        cls.db = cls.client[settings.mongodb_name]

    @classmethod
    async def disconnect(cls):
        cls.client.close()


# MongoDB dependency
async def get_database():
    return Database.db