import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from .model import get_model
from config import STOCK_REGISTRY
from .utils import setup_logger

logger = setup_logger("backtester")

class BacktestResult:
    def __init__(self, metrics: Dict[str, Any], equity_curve: pd.DataFrame):
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.currency = metrics.get('currency', 'USD')
        self.currency_symbol = metrics.get('currency_symbol', '$')
        self.benchmark_name = metrics.get('benchmark_name', 'Benchmark')

def run_backtest(X: pd.DataFrame, y: pd.Series, model_name: str, ticker: str, initial_capital: float = 10000.0, transaction_cost: float = 0.001) -> BacktestResult:
    """Walk-forward backtesting."""
    logger.info(f"Running backtest for {ticker} using {model_name}")
    tscv = TimeSeriesSplit(n_splits=5)
    
    pipeline = get_model(model_name)
    
    all_preds = []
    all_indices = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        all_preds.extend(preds)
        all_indices.extend(test_index)
    
    # Evaluate over all test predictions
    y_test_all = y.iloc[all_indices]
    
    # Equity curve simulation
    df_test = X.iloc[all_indices].copy()
    df_test['pred_signal'] = all_preds
    # Return from Open to Close or Close to next Close
    # Since we predict if future close > today's close, the return is tomorrow's pct_change
    df_test['stock_return'] = df_test['Close'].pct_change().shift(-1).fillna(0)
    
    df_test['strategy_return'] = df_test['stock_return'] * np.where(df_test['pred_signal'] == 1, 1, -1)
    
    # Transaction costs applied when signal changes
    df_test['signal_change'] = df_test['pred_signal'].diff().fillna(0) != 0
    df_test.loc[df_test['signal_change'], 'strategy_return'] -= transaction_cost
    
    df_test['strategy_equity'] = initial_capital * (1 + df_test['strategy_return']).cumprod()
    df_test['buy_hold_equity'] = initial_capital * (1 + df_test['stock_return']).cumprod()
    
    # Metrics
    strategy_returns = df_test['strategy_return']
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)
    win_rate = np.mean(strategy_returns > 0)
    
    cum_ret = df_test['strategy_equity'].iloc[-1] / initial_capital - 1
    max_drawdown = ((df_test['strategy_equity'].cummax() - df_test['strategy_equity']) / df_test['strategy_equity'].cummax()).max()
    
    currency = "USD"
    currency_symbol = "$"
    benchmark_name = "S&P 500"
    if ticker in STOCK_REGISTRY:
        conf = STOCK_REGISTRY[ticker]
        currency = conf.currency
        currency_symbol = conf.currency_symbol
        benchmark_name = conf.benchmark
    else:
        from .utils import get_currency_info, get_benchmark_for_ticker
        currency, currency_symbol = get_currency_info(ticker)
        benchmark_name = get_benchmark_for_ticker(ticker)
        
    metrics = {
        "sharpe": sharpe,
        "win_rate": win_rate,
        "cumulative_return": cum_ret,
        "max_drawdown": max_drawdown,
        "currency": currency,
        "currency_symbol": currency_symbol,
        "benchmark_name": benchmark_name
    }
    
    return BacktestResult(metrics, df_test)
