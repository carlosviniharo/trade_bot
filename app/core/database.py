import os

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
from pymongo.errors import ConnectionFailure


# Load settings from environment variables
class Settings(BaseSettings):
    mongodb_uri: str
    mongodb_name: str

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), '../../.env')


settings = Settings()


class Database:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        cls.client = AsyncIOMotorClient(settings.mongodb_uri)
        cls.db = cls.client[settings.mongodb_name]
        # Check if the database exists
        if settings.mongodb_name not in await cls.client.list_database_names():
            # Database does not exist, create it and insert an initial document
            await cls.create_initial_document()

    @classmethod
    async def disconnect(cls):
        cls.client.close()

    @classmethod
    async def create_initial_document(cls):
        # Create an initial document in the specified collection
        initial_document = {"init": "This is a sample document to create the database."}
        collection_name = "my_collection"  # Replace with your actual collection name
        await cls.db[collection_name].insert_one(initial_document)


# MongoDB dependency
async def get_database():
    return Database.db