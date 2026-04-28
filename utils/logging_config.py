"""
Logging configuration and utilities.
Provides structured logging for the extraction system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config.settings import config

class ColoredFormatter(logging.Formatter):
    """Custom formatter w/ color codes for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with color."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "paper_extractor",
    log_file: Optional[Path] = None,
    level: str = config.LOG_LEVEL
    ) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file. If None, uses config default
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = config.LOG_FILE
    
    log_file = log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_extraction_start(logger: logging.Logger, pdf_path: str) -> None:
    """Log the start of an extraction process."""
    logger.info(f"=" * 50)
    logger.info(f"Starting extraction for: {pdf_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"=" * 50)


def log_extraction_complete(
    logger: logging.Logger,
    pdf_path: str,
    processing_time: float,
    success: bool = True
    ) -> None:
    """Log the completion of an extraction process."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"=" * 50)
    logger.info(f"Extraction {status} for: {pdf_path}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"=" * 50)


def log_model_response(
    logger: logging.Logger,
    model_name: str,
    provider: str,
    success: bool,
    processing_time: Optional[float] = None,
    error: Optional[str] = None
    ) -> None:
    """Log a model's response."""
    status = "✓" if success else "x"
    time_str = f" ({processing_time:.2f}s)" if processing_time else ""
    
    if success:
        logger.info(f"{status} {provider}/{model_name}{time_str}")
    else:
        logger.error(f"{status} {provider}/{model_name}{time_str} - Error: {error}")


# Create default logger instance
logger = setup_logger()
