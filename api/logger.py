import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = "logs/api.log"

def get_logger():
    logger = logging.getLogger("domain_intelligence_api")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes = 2_000_000,
            backupCount = 3
            )

    formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
            )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
