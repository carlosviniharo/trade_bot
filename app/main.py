import asyncio
import sys

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.task.scheduler import start_scheduler, shutdown_scheduler
from app.routers import trade_router
from app.core.database import Database
from app.core.logging import AppLogger

# Set up logging
logger = AppLogger.get_logger()

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set Windows event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        logger.info("Connecting to the database...")
        await Database.connect()  # Connect to the database
        logger.info("Database connected.")

        start_scheduler()  # Start the scheduler

        yield

    except Exception as e:
        logger.error(f"Error during lifespan: {e}")

    finally:
        shutdown_scheduler()  # Shutdown the scheduler
        logger.info("Disconnecting from the database...")
        await Database.disconnect()  # Disconnect from the database
        logger.info("Database disconnected.")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Example route
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the FastAPI app!"}


# Include routers
app.include_router(trade_router.router)


