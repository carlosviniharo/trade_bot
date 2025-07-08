import asyncio
import sys
from fastapi import HTTPException
from apscheduler.triggers.cron import CronTrigger

from app.models.market_models import MarketEvent

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

        top_price_increase = analyzer.get_top_symbols(metric="price_change")
        top_price_decrease = analyzer.get_top_symbols(metric="price_change", ascending=False)
        top_volume_change = analyzer.get_top_symbols(metric="volume_change")

        if any([top_price_increase, top_price_decrease, top_volume_change]):
            telegram = TelegramOutput(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)

            # Prepare event dictionaries
            top_price_increase_v = [MarketEvent(**event).model_dump() for event in top_price_increase]
            top_price_decrease_v = [MarketEvent(**event).model_dump() for event in top_price_decrease]
            top_volume_change_v = [MarketEvent(**event).model_dump() for event in top_volume_change]

            # Insert non-empty lists and log
            if top_price_increase_v:
                await db["market_events"].insert_many(top_price_increase_v)
                logger.info(f"Inserted {len(top_price_increase_v)} positive price change records.")
            else:
                logger.info("No positive price change records to insert.")

            if top_price_decrease_v:
                await db["market_events"].insert_many(top_price_decrease_v)
                logger.info(f"Inserted {len(top_price_decrease_v)} negative price change records.")
            else:
                logger.info("No negative price change records to insert.")

            if top_volume_change_v:
                await db["market_events"].insert_many(top_volume_change_v)
                logger.info(f"Inserted {len(top_volume_change_v)} volume change records.")
            else:
                logger.info("No volume change records to insert.")

            # Combine all for message formatting
            message = format_message_spikes(
                *top_price_increase, *top_price_decrease, *top_volume_change
            )

            if message:
                try:
                    await telegram.send_text_message(message)
                except Exception as e:
                    logger.exception(f"Failed to send message to Telegram: {e}")
                    raise HTTPException(
                        status_code=502, detail=f"Telegram delivery failed: {str(e)}"
                    )
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
    logger.info("Starting scheduler...")
    trigger = CronTrigger(minute="13,28,43,58")
    # trigger = CronTrigger(minute="*/2")
    scheduler.add_job(scheduled_task, trigger)
    scheduler.start()

def shutdown_scheduler():
    logger.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")