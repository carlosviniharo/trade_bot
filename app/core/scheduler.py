import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import time
from app.core.database import get_database
from app.models.trade import TradeCreate
from app.utils.helper import calculate_volume_changes

# Initialize logging
logging.basicConfig(level=logging.INFO)

async def scheduled_task():
    try:
        logging.info("Scheduled task started")
        db = await get_database()

        trade_data = await calculate_volume_changes()

        if isinstance(trade_data, list) and len(trade_data) > 0:

            # Validate each item and create a list of Pydantic models
            trade_models = [TradeCreate(**trade) for trade in trade_data]
            # Convert each Pydantic model back to a dictionary before inserting into MongoDB
            valid_trade_data = [trade.model_dump() for trade in trade_models]
            # Insert the list of validated dictionaries into MongoDB
            await db["trades"].insert_many(valid_trade_data)

        else:
            logging.info("No trade data to insert")

    except Exception as e:
        logging.error(f"An error occurred: {e}")



def run_async_task():
    try:
        loop = asyncio.get_running_loop()  # Try to get the current running loop
    except RuntimeError:
        loop = asyncio.new_event_loop()  # Create a new loop if none exists
        asyncio.set_event_loop(loop)

        # Run the async task and wait for it to complete
    loop.run_until_complete(scheduled_task())

# Initialize the scheduler
scheduler = AsyncIOScheduler()

def start_scheduler():
    # Schedule a job to start at 12:07 and then run every 3 minutes
    logging.info("Starting scheduler...")
    scheduler.add_job(scheduled_task, "interval", minutes=2)  # Correct function reference
    scheduler.start()

def shutdown_scheduler():
    logging.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logging.info("Scheduler shut down.")