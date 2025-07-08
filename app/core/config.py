from pydantic_settings import BaseSettings
from typing import Optional

class AppSettings(BaseSettings):
    ENV: str = "development" 
    APP_NAME: str = "Trade Bot"
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_NAME: str = "mydatabase"
    LOG_LEVEL: str = "INFO"
    WHATSAPP_TOKEN: Optional[str] = None
    PHONE_NUMBER_ID: Optional[str] = None
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    DEBUG: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

class DevelopmentSettings(AppSettings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(AppSettings):
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"

def get_settings():
    """
    Factory function to select the appropriate settings
    based on the ENV environment variable.
    """
    from os import getenv
    env = getenv("ENV", "development")
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    else:
        raise ValueError(f"Unknown environment: {env}")

settings = get_settings()
