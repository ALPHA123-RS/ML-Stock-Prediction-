import argparse
import sys
import pandas as pd
from src.data_loader import fetch_stock_data, get_company_info
from src.features import build_features
from src.model import load_model, predict_today
from src.utils import setup_logger, format_price
from config import STOCK_REGISTRY, DEFAULT_HORIZON

logger = setup_logger("predict")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Specific stock ticker")
    parser.add_argument("--model", type=str, default="xgb", help="Model to use")
    
    args = parser.parse_args()
    
    ticker = args.ticker
    if not ticker:
        ticker = input("Enter ticker to predict: ").strip().upper()
        
    if ticker in STOCK_REGISTRY:
        start_date = STOCK_REGISTRY[ticker].train_start
    else:
        start_date = "2020-01-01"
        
    logger.info(f"Fetching recent data for {ticker}")
    # Need enough data for 63d regime, 200d MA, so fetch from 1 year ago
    end_date = str(pd.Timestamp.today().date())
    start_dt = pd.Timestamp.today() - pd.Timedelta(days=365)
    
    df = fetch_stock_data(ticker, str(start_dt.date()), end_date)
    if df.empty:
        logger.error("No data fetched.")
        sys.exit(1)
        
    X, y = build_features(df, horizon=DEFAULT_HORIZON, ticker=ticker)
    if X.empty:
        logger.error("Feature building failed.")
        sys.exit(1)
        
    try:
        pipeline = load_model(ticker, args.model)
    except FileNotFoundError:
        logger.error(f"Model {args.model} for {ticker} not found. Please train first.")
        sys.exit(1)
        
    result = predict_today(pipeline, X, ticker, horizon_days=DEFAULT_HORIZON)
    
    # Render Output
    info = get_company_info(ticker)
    name = info.get("name", ticker)
    
    currency_sym = STOCK_REGISTRY[ticker].currency_symbol if ticker in STOCK_REGISTRY else '$'
    price_str = format_price(result['price'], currency_sym, result['currency'])
    
    arrow = "▲" if result['direction'] == "UP" else "▼"
    
    print("╔════════════════════════════════════════╗")
    print(f"║  {name} ({ticker})")
    print(f"║  Last price: {price_str}")
    print(f"║  Prediction: {arrow} {result['direction']}")
    print(f"║  Confidence: {result['confidence'] * 100:.1f}% ({result['confidence_label']})")
    print(f"║  Horizon:    {result['horizon_days']} trading day")
    print("╚════════════════════════════════════════╝")
    
    print("\nTop drivers (SHAP):")
    for feat, shap_val in result['top_features']:
        sign = "↑" if shap_val > 0 else "↓"
        print(f"  {sign} {feat:<15} {shap_val:+.3f}")

if __name__ == "__main__":
    main()
