import asyncio
import sys

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from app.core.scheduler import start_scheduler, shutdown_scheduler
from app.routers import trade_router
from app.core.database import Database
from app.core.logging import setup_logging


# Set up logging
setup_logging()

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set Windows event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        logging.info("Connecting to the database...")
        await Database.connect()  # Connect to the database
        logging.info("Database connected.")

        start_scheduler()  # Start the scheduler

        yield

    except Exception as e:
        logging.error(f"Error during lifespan: {e}")

    finally:
        shutdown_scheduler()  # Shutdown the scheduler
        logging.info("Disconnecting from the database...")
        await Database.disconnect()  # Disconnect from the database
        logging.info("Database disconnected.")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Example route
@app.get("/")
async def read_root():
    logging.info("Root endpoint accessed.")
    return {"message": "Welcome to the FastAPI app!"}


# Include routers
app.include_router(trade_router.router)


