from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings



class Database:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        cls.client = AsyncIOMotorClient(settings.MONGODB_URI)
        cls.db = cls.client[settings.MONGODB_NAME]
        # Check if the database exists
        if settings.MONGODB_NAME not in await cls.client.list_database_names():
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
