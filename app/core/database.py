from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings



class Database:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URI)
            cls.db = cls.client[settings.MONGODB_NAME]
            # Ping the server to confirm connectivity (SRV DNS resolved here)
            await cls.client.admin.command("ping")
            # Check if the database exists; create it if not
            if settings.MONGODB_NAME not in await cls.client.list_database_names():
                await cls.create_initial_document()
        except Exception as e:
            cls.client = None
            cls.db = None
            raise RuntimeError(f"Failed to connect to MongoDB: {e}") from e

    @classmethod
    async def disconnect(cls):
        if cls.client is not None:
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
