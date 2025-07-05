import asyncio
import sys
from fastapi import HTTPException
from apscheduler.triggers.cron import CronTrigger

# Set the event loop policy for Windows if necessary
if sys.platform == 'win32':
    # Check if aiodns is imported, and apply the event loop policy
    try:
        import aiodns
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except ImportError:
        pass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.core.database import get_database
from app.core.logging import AppLogger
from app.models.trade import StockChangeRecordCreate
from app.utils.helper import BinanceVolumeAnalyzer, format_message_spikes
from app.utils.whatsapp_connector import WhatsAppOutput
from app.utils.telegram_connector import TelegramOutput
from app.core.config import settings

# Initialize logging
logger = AppLogger.get_logger()

async def scheduled_task():
    analyzer = BinanceVolumeAnalyzer()

    try:
        logger.info("Scheduled task started")
        db = await get_database()

        await analyzer.initialize()
        await analyzer.calculate_market_spikes()
        most_price_change = analyzer.get_top_symbols(metric="price_change")
        less_price_change = analyzer.get_top_symbols(metric="price_change", ascending=False)
        most_volume_change = analyzer.get_top_symbols(metric="volume_change")

        # sym = [{'symbol': 'MOVE', 'price_change': 3.5574720388087844, 'close': 0.7685},
        #        {'symbol': 'ACX', 'price_change': -2.0331030887527635, 'close': 0.7517},
        #        {'symbol': 'NULS', 'price_change': -3.8431217973984966, 'close': 0.4879}]
        #
        # sym_v = [{'symbol': 'NULS', 'volume_change': 35.93, 'close': 0.4879},
        #          {'symbol': 'ACX', 'volume_change': 17.59, 'close': 0.7517},
        #          {'symbol': 'MOVE', 'volume_change': 16.94, 'close': 0.7685}]

        if len(most_price_change) > 0 or len(most_volume_change) > 0:

            # whatsapp = WhatsAppOutput(settings.WHATSAPP_TOKEN, settings.PHONE_NUMBER_ID)
            telegram = TelegramOutput(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)

            # Validate each item and create a list of Pydantic models
            stock_change_record = StockChangeRecordCreate(
                price_changes=most_price_change,
                volume_changes=most_volume_change
            )
            # Convert each Pydantic model back to a dictionary before inserting into MongoDB
            await db["stock_change_records"].insert_one(stock_change_record.model_dump())
            logger.info("Trade data inserted")

            message = format_message_spikes(*most_price_change, *most_volume_change)

            if message:
                # await whatsapp.send_text_message("447729752680", message)
                try:
                    await telegram.send_text_message(message)
                except Exception as e:
                    logger.exception(f"Failed to send message to Telegram: {e}")
                    raise HTTPException(status_code=502, detail=f"Telegram delivery failed: {str(e)}")
                finally:
                    await telegram.close()
        else:
            logger.info("No trade data to insert")

    except Exception as e:

        logger.error(f"An error occurred: {e}")

    finally:
        await analyzer.close()


scheduler = AsyncIOScheduler()
# loop = asyncio.get_event_loop()

def start_scheduler():
    # Schedule a job to start at 12:07 and then run every 3 minutes
    logger.info("Starting scheduler...")
    trigger = CronTrigger(minute="13,28,43,58")
    # trigger = CronTrigger(minute="*/2")
    scheduler.add_job(scheduled_task, trigger)
    # scheduler.add_job(scheduled_task, "interval", minutes=2)
    scheduler.start()

def shutdown_scheduler():
    logger.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")