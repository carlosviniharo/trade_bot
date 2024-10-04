from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.routers import trade_router
from app.core.database import Database  # Assuming Database class is here

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Connect to the database
    await Database.connect()
    yield  # Hand over control to FastAPI routes
    # Shutdown logic: Disconnect from the database
    await Database.disconnect()

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)


# Example route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Include routers
app.include_router(trade_router.router)

#
# # Endpoint to test MongoDB connection
# @app.get("/test-connection")
# async def test_connection():
#     try:
#         # Attempt to fetch the server status
#         server_info = await Database.db.command("ping")
#         return {"status": "Connected", "server_info": server_info}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")

