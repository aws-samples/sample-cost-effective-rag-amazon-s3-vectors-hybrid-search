"""Minimal structured (JSON) logging for CloudWatch."""
import json
import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logger


def log_json(logger: logging.Logger, message: str, /, level: int = logging.INFO, **fields) -> None:
    """Emit a single-line JSON log record (easy to query in Logs Insights)."""
    logger.log(level, json.dumps({"message": message, **fields}, default=str))
