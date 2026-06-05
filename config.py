import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class StockConfig:
    ticker: str
    name: str
    exchange: str
    currency: str
    currency_symbol: str
    sector: str
    train_start: str
    benchmark: str
    description: str

STOCK_REGISTRY: Dict[str, StockConfig] = {
    "NVDA": StockConfig(
        ticker        = "NVDA",
        name          = "NVIDIA Corporation",
        exchange      = "NASDAQ",
        currency      = "USD",
        currency_symbol = "$",
        sector        = "Semiconductors",
        train_start   = "2018-01-01",
        benchmark     = "^GSPC",
        description   = "AI chip giant powering the LLM revolution"
    ),
    "GOOGL": StockConfig(
        ticker        = "GOOGL",
        name          = "Alphabet Inc.",
        exchange      = "NASDAQ",
        currency      = "USD",
        currency_symbol = "$",
        sector        = "Technology",
        train_start   = "2018-01-01",
        benchmark     = "^GSPC",
        description   = "Google's parent: search, cloud, and AI"
    ),
    "NKE": StockConfig(
        ticker        = "NKE",
        name          = "Nike Inc.",
        exchange      = "NYSE",
        currency      = "USD",
        currency_symbol = "$",
        sector        = "Consumer Goods",
        train_start   = "2018-01-01",
        benchmark     = "^GSPC",
        description   = "Global sportswear giant undergoing turnaround"
    ),
    "TATASTEEL.NS": StockConfig(
        ticker        = "TATASTEEL.NS",
        name          = "Tata Steel Ltd",
        exchange      = "NSE",
        currency      = "INR",
        currency_symbol = "₹",
        sector        = "Materials",
        train_start   = "2018-01-01",
        benchmark     = "^NSEI",
        description   = "India's leading integrated steel manufacturer"
    ),
}

DEFAULT_MODELS: List[str] = ["lr", "rf", "xgb", "lgbm", "ensemble"]
DEFAULT_HORIZON: int = 1
TEST_SIZE: float = 0.20
TRANSACTION_COST: float = 10.0  # basis points
INITIAL_CAPITAL: float = 10_000.0
RANDOM_SEED: int = 42
CV_FOLDS: int = 5
OPTUNA_TRIALS: int = 50

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "data" / "cache"

for d in [MODELS_DIR, REPORTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
