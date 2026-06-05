import logging
import sys
from babel.numbers import format_currency
from typing import Tuple

def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def format_price(value: float, currency_symbol: str, currency_code: str) -> str:
    """
    Formats price based on currency.
    For INR, uses the Indian numbering system if applicable.
    """
    if value is None:
        return "N/A"
    
    if currency_code == "INR":
        if value >= 1_00_00_000: # Crores
            val = value / 1_00_00_000
            return f"₹{val:.2f} Cr"
        elif value >= 1_00_000: # Lakhs
            val = value / 1_00_000
            return f"₹{val:.2f} L"
        else:
            return format_currency(value, 'INR', locale='en_IN')
    else:
        return format_currency(value, currency_code, locale='en_US')

def get_currency_info(ticker: str) -> Tuple[str, str]:
    """Helper to guess currency if not in config."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "INR", "₹"
    return "USD", "$"

def get_benchmark_for_ticker(ticker: str) -> str:
    """Helper to guess benchmark if not in config."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "^NSEI"
    return "^GSPC"
