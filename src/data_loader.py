import pandas as pd
import numpy as np
import yfinance as yf
import hashlib
import time
import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from .utils import setup_logger
from config import CACHE_DIR, STOCK_REGISTRY

logger = setup_logger("data_loader")

def _get_cache_path(ticker: str, start: str, end: str) -> Path:
    key = f"{ticker}_{start}_{end}".encode("utf-8")
    hash_key = hashlib.md5(key).hexdigest()
    return CACHE_DIR / f"{ticker}_{hash_key}.parquet"

def fetch_stock_data(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetches stock data with caching, retries, and validation."""
    cache_path = _get_cache_path(ticker, start, end)
    
    if use_cache and cache_path.exists():
        logger.info(f"Loading {ticker} from cache.")
        df = pd.read_parquet(cache_path)
        return df

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {ticker} from {start} to {end} (Attempt {attempt+1})")
            # Set auto_adjust=True for splits/dividends
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            
            if df.empty:
                logger.warning(f"No data fetched for {ticker}")
                return pd.DataFrame()

            # Flatten multi-index if necessary (yfinance behavior)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
            df = df.reset_index()
            
            # Normalize timezone to tz-naive UTC
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)

            # Data quality gates
            # 1. No future dates
            today = pd.Timestamp.utcnow().tz_localize(None)
            df = df[df['Date'] <= today]
            
            # 2. No negative/zero prices
            df = df[(df['Close'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Open'] > 0)]
            
            # 3. Flag volume spikes > 10 sigma (we don't drop, just flag or cap)
            if len(df) > 20:
                vol_mean = df['Volume'].rolling(20).mean()
                vol_std = df['Volume'].rolling(20).std()
                vol_spike_threshold = vol_mean + 10 * vol_std
                # Clip extreme volume spikes to avoid feature distortion
                df.loc[df['Volume'] > vol_spike_threshold, 'Volume'] = vol_spike_threshold

            df.to_parquet(cache_path)
            logger.info(f"Successfully fetched and cached {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed for {ticker}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def get_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Returns daily log returns of the benchmark aligned to stock calendar."""
    if ticker in STOCK_REGISTRY:
        bench_ticker = STOCK_REGISTRY[ticker].benchmark
    else:
        from .utils import get_benchmark_for_ticker
        bench_ticker = get_benchmark_for_ticker(ticker)
        
    df_bench = fetch_stock_data(bench_ticker, start, end)
    if df_bench.empty:
        return pd.Series()
    
    df_bench.set_index('Date', inplace=True)
    df_bench['bench_return'] = np.log(df_bench['Close']) - np.log(df_bench['Close'].shift(1))
    return df_bench['bench_return']

def get_company_info(ticker: str) -> Dict[str, Any]:
    """Returns basic company info with fallback."""
    if ticker in STOCK_REGISTRY:
        conf = STOCK_REGISTRY[ticker]
        return {
            "name": conf.name,
            "sector": conf.sector,
            "country": "India" if conf.currency == "INR" else "USA",
            "market_cap": None,
            "currency": conf.currency,
            "exchange": conf.exchange,
            "description": conf.description
        }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        from .utils import get_currency_info
        curr, sym = get_currency_info(ticker)
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "country": info.get("country", "Unknown"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", curr),
            "exchange": info.get("exchange", "Unknown"),
            "description": info.get("longBusinessSummary", "No description available.")
        }
    except Exception:
        from .utils import get_currency_info
        curr, sym = get_currency_info(ticker)
        return {
            "name": ticker,
            "sector": "Unknown",
            "country": "Unknown",
            "market_cap": None,
            "currency": curr,
            "exchange": "Unknown",
            "description": "No description available."
        }
