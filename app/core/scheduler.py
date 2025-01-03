import asyncio
import sys

from apscheduler.triggers.cron import CronTrigger

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.core.database import get_database
from app.core.logging import AppLogger
from app.models.trade import StockChangeRecordCreate
from app.utils.helper import BinanceVolumeAnalyzer

# Initialize logging
logger = AppLogger.get_logger()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def scheduled_task():
    analyzer = BinanceVolumeAnalyzer()

    try:
        logger.info("Scheduled task started")
        db = await get_database()

        await analyzer.initialize()
        await analyzer.calculate_market_spikes()
        most_price_change = analyzer.get_best_symbols("price_change")
        most_volume_change = analyzer.get_best_symbols("volume_change")


        if len(most_price_change) > 0 or len(most_volume_change) > 0:

            # Validate each item and create a list of Pydantic models
            stock_change_record = StockChangeRecordCreate(
                price_changes=most_price_change,
                volume_changes=most_volume_change
            )
            # Convert each Pydantic model back to a dictionary before inserting into MongoDB
            await db["stock_change_records"].insert_one(stock_change_record.model_dump())
            logger.info("Trade data inserted")

        else:
            logger.info("No trade data to insert")

    except Exception as e:

        logger.error(f"An error occurred: {e}")

    finally:
        await analyzer.close()


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
    logger.info("Starting scheduler...")
    trigger = CronTrigger(minute="13,28,43,58")
    scheduler.add_job(scheduled_task, trigger)
    # scheduler.add_job(scheduled_task, "interval", minutes=2)
    scheduler.start()

def shutdown_scheduler():
    logger.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")