from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

# Function that you want to run periodically
def scheduled_task():
    print(f"Task executed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize the scheduler
scheduler = BackgroundScheduler()

def start_scheduler():
    # Schedule a job using a cron expression
    # Schedule a job to start at 12:07 and then run every 15 minutes
    scheduler.add_job(scheduled_task, CronTrigger(hour='*', minute='7,22,37,52'))
    scheduler.start()

def shutdown_scheduler():
    scheduler.shutdown()