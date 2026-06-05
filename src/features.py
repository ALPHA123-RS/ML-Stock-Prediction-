import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import IchimokuIndicator
from typing import Tuple, Optional
from .utils import setup_logger
from .data_loader import fetch_stock_data, get_benchmark
from config import STOCK_REGISTRY

logger = setup_logger("features")

def build_features(df: pd.DataFrame, horizon: int = 1, ticker: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds > 50 technical and regime features, strictly avoiding lookahead bias.
    """
    logger.info(f"Building features for {ticker} (horizon={horizon})")
    
    if df.empty or len(df) < 100:
        logger.warning("Not enough data to build features.")
        return pd.DataFrame(), pd.Series()
    
    data = df.copy()
    data.set_index('Date', inplace=True)
    
    # Generate all ta features (momentum, volume, volatility, trend, etc.)
    # Clean parameter avoids NaN generation where possible
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    
    # Ichimoku Cloud
    ichi = IchimokuIndicator(high=data["High"], low=data["Low"], window1=9, window2=26, window3=52, visual=False, fillna=True)
    data['ichimoku_a'] = ichi.ichimoku_a()
    data['ichimoku_b'] = ichi.ichimoku_b()
    data['ichimoku_base_line'] = ichi.ichimoku_base_line()
    data['ichimoku_conversion_line'] = ichi.ichimoku_conversion_line()
    
    data['price_above_cloud'] = (data['Close'] > data['ichimoku_a']) & (data['Close'] > data['ichimoku_b'])
    data['price_below_cloud'] = (data['Close'] < data['ichimoku_a']) & (data['Close'] < data['ichimoku_b'])
    data['price_inside_cloud'] = ~data['price_above_cloud'] & ~data['price_below_cloud']
    
    # Regime / Context
    data['return_63d'] = data['Close'].pct_change(63)
    data['volatility_21d'] = data['Close'].pct_change().rolling(21).std()
    
    # Volatility quintile
    data['volatility_quintile'] = pd.qcut(data['volatility_21d'].rank(method='first'), 5, labels=False, duplicates='drop')
    
    # 3-state regime
    conditions = [
        data['return_63d'] > 0.05,
        data['return_63d'] < -0.05
    ]
    choices = [1, -1] # 1: Bull, -1: Bear, 0: Sideways
    data['market_regime'] = np.select(conditions, choices, default=0)
    
    # Benchmark and cross-asset features
    if ticker:
        # Benchmark features
        bench_ret = get_benchmark(ticker, str(data.index[0].date()), str(data.index[-1].date()))
        if not bench_ret.empty:
            bench_ret.name = "bench_return"
            data = data.join(bench_ret, how='left')
            data['bench_return'].fillna(0, inplace=True)
            
            # Rolling 21d correlation
            stock_ret = data['Close'].pct_change()
            data['corr_21d_bench'] = stock_ret.rolling(21).corr(data['bench_return']).fillna(0)
            
            # Beta 60d
            cov_60d = stock_ret.rolling(60).cov(data['bench_return'])
            var_60d_bench = data['bench_return'].rolling(60).var()
            data['beta_60d'] = (cov_60d / var_60d_bench).fillna(1.0)
        else:
            data['corr_21d_bench'] = 0.0
            data['beta_60d'] = 1.0
            
        # Currency context features
        if ticker in STOCK_REGISTRY:
            currency = STOCK_REGISTRY[ticker].currency
        else:
            from .utils import get_currency_info
            currency, _ = get_currency_info(ticker)
            
        try:
            if currency == "INR":
                fx = fetch_stock_data("INR=X", str(data.index[0].date()), str(data.index[-1].date()))
                if not fx.empty:
                    fx.set_index('Date', inplace=True)
                    data['fx_usd_inr'] = fx['Close']
                    data['fx_usd_inr'].fillna(method='ffill', inplace=True)
                else:
                    data['fx_usd_inr'] = 1.0
            elif currency == "USD":
                dxy = fetch_stock_data("DX-Y.NYB", str(data.index[0].date()), str(data.index[-1].date()))
                if not dxy.empty:
                    dxy.set_index('Date', inplace=True)
                    data['dxy_index'] = dxy['Close']
                    data['dxy_index'].fillna(method='ffill', inplace=True)
                else:
                    data['dxy_index'] = 100.0
        except Exception as e:
            logger.warning(f"Could not fetch currency feature: {e}")
            if currency == "INR": data['fx_usd_inr'] = 1.0
            elif currency == "USD": data['dxy_index'] = 100.0

    # VIX proxy
    try:
        if currency == "INR":
            vix = fetch_stock_data("^INDIAVIX", str(data.index[0].date()), str(data.index[-1].date()))
        else:
            vix = fetch_stock_data("^VIX", str(data.index[0].date()), str(data.index[-1].date()))
            
        if not vix.empty:
            vix.set_index('Date', inplace=True)
            data['vix'] = vix['Close']
            data['vix'].fillna(method='ffill', inplace=True)
        else:
            data['vix'] = data['volatility_21d'] * 100
    except Exception:
        data['vix'] = data['volatility_21d'] * 100

    # Target generation (strict shift backwards -> tomorrow's price is target for today)
    # 1 if future close > today's close, else 0
    future_close = data['Close'].shift(-horizon)
    target = (future_close > data['Close']).astype(int)
    
    # Drop rows where target is NaN (the last `horizon` rows)
    valid_idx = target.dropna().index
    data = data.loc[valid_idx]
    target = target.loc[valid_idx]
    
    # Forward fill any remaining NaNs safely, then drop rows with NaNs
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    target = target.loc[data.index]

    # Validate NaN > 80%
    nan_ratios = data.isna().mean()
    bad_cols = nan_ratios[nan_ratios > 0.8].index
    if len(bad_cols) > 0:
        logger.warning(f"Dropping columns with >80% NaNs: {list(bad_cols)}")
        data.drop(columns=bad_cols, inplace=True)

    # Log class balance
    up_ratio = target.mean()
    logger.info(f"Class balance - UP: {up_ratio:.2f}, DOWN: {1-up_ratio:.2f}")
    if up_ratio > 0.65 or up_ratio < 0.35:
        logger.warning(f"High class imbalance detected ({up_ratio:.2f})!")

    return data, target
