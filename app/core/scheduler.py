import asyncio
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time
from app.core.database import get_database
from app.utils.helper import calculate_volume_changes


# Function that you want to run periodically
async def scheduled_task():
    # db = await get_database()
    # trade_data = await calculate_volume_changes()
    # if trade_data is not {}:
    #     await db["trades"].insert_one(trade_data[0])
    logging.info(f"Task executed at {time.strftime('%Y-%m-%d %H:%M:%S')}")


def run_async_task():
    loop = asyncio.get_event_loop()  # Get the current event loop
    if loop.is_running():
        # If the loop is already running, use create_task instead of asyncio.run()
        loop.create_task(scheduled_task())
    else:
        # If no loop is running, start one with asyncio.run()
        asyncio.run(scheduled_task())

# Initialize the scheduler
scheduler = BackgroundScheduler()

def start_scheduler():
    # Schedule a job to start at 12:07 and then run every 15 minutes
    logging.info("Starting scheduler...")
    # scheduler.add_job(scheduled_task, CronTrigger(hour='*', minute='7,22,37,44,49,50,51,52'))
    scheduler.add_job(scheduled_task, CronTrigger.from_crontab('*/4 * * * *'))
    scheduler.start()

def shutdown_scheduler():
    logging.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logging.info("Scheduler shut down.")