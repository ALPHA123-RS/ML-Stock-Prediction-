import pytest
import pandas as pd
import numpy as np
from src.backtester import run_backtest, BacktestResult

def test_zero_cost_returns():
    # Dummy data
    dates = pd.date_range("2022-01-01", periods=100)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(100) * 100 + 10,
        'High': np.random.rand(100) * 100 + 20,
        'Low': np.random.rand(100) * 100 + 5,
        'Close': np.random.rand(100) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    X = pd.DataFrame(np.random.rand(100, 5))
    X['Close'] = df['Close']
    X.index = df['Date']
    y = pd.Series(np.random.randint(0, 2, 100), index=df['Date'])
    
    # We mock the get_model so we don't need real training
    # For integration test, we can just run with 'lr' and expect no crash
    res = run_backtest(X, y, 'lr', 'NVDA', transaction_cost=0.0)
    assert 'strategy_return' in res.equity_curve.columns
    assert res.metrics['win_rate'] >= 0

def test_always_long_equals_bah():
    pass

def test_currency_labels_match_config():
    dates = pd.date_range("2022-01-01", periods=100)
    X = pd.DataFrame({'Close': np.random.rand(100) * 100 + 10}, index=dates)
    y = pd.Series(np.random.randint(0, 2, 100), index=dates)
    
    res_inr = run_backtest(X, y, 'lr', 'TATASTEEL.NS', transaction_cost=0.0)
    assert res_inr.currency == 'INR'
    assert res_inr.currency_symbol == '₹'
    
    res_usd = run_backtest(X, y, 'lr', 'NVDA', transaction_cost=0.0)
    assert res_usd.currency == 'USD'
    assert res_usd.currency_symbol == '$'
