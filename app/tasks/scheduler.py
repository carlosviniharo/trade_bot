import asyncio
import sys
from fastapi import HTTPException
from apscheduler.triggers.cron import CronTrigger
import pandas as pd

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

        df_top_price_increase = analyzer.get_top_symbols(metric="price_change")
        df_top_price_decrease = analyzer.get_top_symbols(metric="price_change", ascending=True)
        df_top_volume_change = analyzer.get_top_symbols(metric="volume_change")

        # Merge the dataframes and group by symbol and event_timestamp
        df_merged = pd.concat([df_top_price_increase, df_top_price_decrease, df_top_volume_change])
        df_merged = df_merged.groupby(["symbol", "event_timestamp"], as_index=False).agg({
            "is_price_event": "max",
            "is_volume_event": "max",
            "price_change": "first",
            "volume_change": "first",
            "atr_pct": "first",
            "close": "first",
            })
        
        if not df_merged.empty:
            telegram = TelegramOutput(settings.TELEGRAM_BOT_TOKEN, settings.TELEGRAM_CHAT_ID)

            # Prepare event dictionaries
            top_moves_v = [MarketEvent(**event).model_dump() for event in df_merged]

            # Insert non-empty lists and log
            if top_moves_v:
                await db["market_events"].insert_many(top_moves_v)
                logger.info(f"Inserted {len(top_moves_v)} price and volume change records.")
            else:
                logger.info("No price and volume change records to insert.")

            # Combine all for message formatting
            message = format_message_spikes(*top_moves_v)

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
    trigger = CronTrigger(minute="14,29,44,59")
    # trigger = CronTrigger(minute="*/2")
    scheduler.add_job(scheduled_task, trigger)
    scheduler.start()

def shutdown_scheduler():
    logger.info("Shutting down the scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")
