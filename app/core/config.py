import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class BaseConfig:
    """Base configuration shared by all environments."""
    APP_NAME: str = "Trade Bot"
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_NAME: str = os.getenv("MONGODB_NAME", "mydatabase")
    LOG_LEVEL: str = "INFO"
    WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN")
    PHONE_NUMBER_ID: str = os.getenv("PHONE_NUMBER_ID")


class DevelopmentConfig(BaseConfig):
    """Development-specific configuration."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


class ProductionConfig(BaseConfig):
    """Production-specific configuration."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"

def get_config():
    """
    Factory function to select the appropriate configuration
    based on the ENV environment variable.
    """
    env = os.getenv("ENV", "development")
    if env == "development":
        return DevelopmentConfig()
    elif env == "production":
        return ProductionConfig()
    else:
        raise ValueError(f"Unknown environment: {env}")

# Singleton configuration instance
settings = get_config()