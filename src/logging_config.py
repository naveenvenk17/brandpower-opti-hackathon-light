"""
Simplified logging configuration using loguru.
Production-ready with minimal setup.
"""
import sys
from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging with loguru.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import os
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper(),
        colorize=True
    )
    
    # Optionally add file handler (create logs dir if needed)
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info(f"Logging configured: level={level}")


def get_logger(name: str):
    """
    Get a logger instance. With loguru, this returns the same logger.
    
    Args:
        name: Module name (for context)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
