import logging
from typing import Optional
from app.core.config import settings


class AppLogger:
    _logger: Optional[logging.Logger] = None  # Use logging.Logger directly

    @classmethod
    def get_logger(cls) -> logging.Logger:  # Use logging.Logger directly
        if cls._logger is None:
            # Set up the logger
            cls._logger = logging.getLogger("AppLogger")
            cls._logger.setLevel(settings.LOG_LEVEL)

            # Create a formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # File handler
            file_handler = logging.FileHandler("app.log", mode="a")
            file_handler.setFormatter(formatter)

            # Add handlers to the logger
            cls._logger.addHandler(console_handler)
            cls._logger.addHandler(file_handler)

        return cls._logger