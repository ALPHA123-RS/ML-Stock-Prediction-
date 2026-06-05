import argparse
import sys
import pandas as pd
from pathlib import Path
from src.data_loader import fetch_stock_data, get_company_info
from src.features import build_features
from src.model import train_model, get_model
from src.backtester import run_backtest
from src.utils import setup_logger
from config import STOCK_REGISTRY, DEFAULT_MODELS, DEFAULT_HORIZON, MODELS_DIR

logger = setup_logger("train")

def interactive_menu():
    print("════════════════════════════════════════")
    print("  Stock Prediction System — Stock Selector")
    print("════════════════════════════════════════")
    print(" [1] NVDA   — NVIDIA Corporation       (NASDAQ · USD)")
    print(" [2] GOOGL  — Alphabet Inc.            (NASDAQ · USD)")
    print(" [3] NKE    — Nike Inc.                (NYSE   · USD)")
    print(" [4] TATASTEEL.NS — Tata Steel Ltd     (NSE    · INR)")
    print(" [5] Enter custom ticker")
    print("════════════════════════════════════════")
    
    choice = input("Select a stock [1-5]: ").strip()
    
    mapping = {
        "1": "NVDA",
        "2": "GOOGL",
        "3": "NKE",
        "4": "TATASTEEL.NS"
    }
    
    if choice in mapping:
        return mapping[choice]
    elif choice == "5":
        return input("Enter custom ticker (e.g., AAPL): ").strip().upper()
    else:
        print("Invalid choice.")
        sys.exit(1)

def show_info_card(ticker: str):
    info = get_company_info(ticker)
    name = info.get('name', ticker)
    exchange = info.get('exchange', 'Unknown')
    currency = info.get('currency', 'Unknown')
    sector = info.get('sector', 'Unknown')
    desc = info.get('description', '')[:50] + "..."
    
    if ticker in STOCK_REGISTRY:
        conf = STOCK_REGISTRY[ticker]
        benchmark = conf.benchmark
        start = conf.train_start
    else:
        from src.utils import get_benchmark_for_ticker
        benchmark = get_benchmark_for_ticker(ticker)
        start = "2018-01-01"
        
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│  {name} ({ticker})")
    print(f"│  Exchange: {exchange}  |  Currency: {currency}  |  Sector: {sector}")
    print(f"│  Benchmark: {benchmark}")
    print(f"│  Training from: {start}")
    print(f"│  Description: {desc}")
    print("└─────────────────────────────────────────────────────┘")
    
    proceed = input(f"Proceed with {ticker}? [Y/n]: ").strip().lower()
    if proceed == 'n':
        sys.exit(0)

def train_pipeline(ticker: str, horizon: int, tune: bool):
    logger.info(f"Starting training pipeline for {ticker}")
    if ticker in STOCK_REGISTRY:
        start_date = STOCK_REGISTRY[ticker].train_start
    else:
        start_date = "2018-01-01"
        
    df = fetch_stock_data(ticker, start_date, str(pd.Timestamp.today().date()))
    if df.empty:
        logger.error(f"Failed to fetch data for {ticker}")
        return
        
    X, y = build_features(df, horizon=horizon, ticker=ticker)
    if X.empty:
        logger.error("Feature building failed.")
        return
        
    for model_name in DEFAULT_MODELS:
        train_model(X, y, model_name, ticker)
        
        # Optionally run backtest to generate report metrics
        res = run_backtest(X, y, model_name, ticker)
        logger.info(f"Backtest {model_name} - Sharpe: {res.metrics['sharpe']:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Specific stock ticker")
    parser.add_argument("--all", action="store_true", help="Train all registry stocks")
    parser.add_argument("--tune", action="store_true", help="With Optuna tuning")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Prediction horizon")
    
    args = parser.parse_args()
    
    if args.all:
        for ticker in STOCK_REGISTRY.keys():
            train_pipeline(ticker, args.horizon, args.tune)
    elif args.ticker:
        train_pipeline(args.ticker, args.horizon, args.tune)
    else:
        ticker = interactive_menu()
        show_info_card(ticker)
        train_pipeline(ticker, args.horizon, args.tune)

if __name__ == "__main__":
    main()
