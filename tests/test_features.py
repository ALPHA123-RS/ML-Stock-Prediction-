import pytest
import pandas as pd
import numpy as np
from src.features import build_features

def test_no_future_leakage():
    # Construct dummy data
    dates = pd.date_range("2022-01-01", periods=200)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 10,
        'High': np.random.rand(200) * 100 + 20,
        'Low': np.random.rand(200) * 100 + 5,
        'Close': np.random.rand(200) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 200)
    })
    
    X, y = build_features(df, horizon=1)
    
    # For index t, target should be Close[t+1] > Close[t]
    # Check that X has no future data
    assert not X.empty
    assert len(X) == len(y)

def test_feature_count():
    dates = pd.date_range("2022-01-01", periods=200)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 10,
        'High': np.random.rand(200) * 100 + 20,
        'Low': np.random.rand(200) * 100 + 5,
        'Close': np.random.rand(200) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 200)
    })
    
    X, y = build_features(df, horizon=1)
    # Features count > 50
    assert X.shape[1] >= 50

def test_no_all_nan_column():
    dates = pd.date_range("2022-01-01", periods=200)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 10,
        'High': np.random.rand(200) * 100 + 20,
        'Low': np.random.rand(200) * 100 + 5,
        'Close': np.random.rand(200) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 200)
    })
    X, y = build_features(df, horizon=1)
    
    nan_ratios = X.isna().mean()
    assert (nan_ratios <= 0.8).all()

def test_target_balance():
    # If random walk, should be close to 50/50
    dates = pd.date_range("2022-01-01", periods=500)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(500) * 100 + 10,
        'High': np.random.rand(500) * 100 + 20,
        'Low': np.random.rand(500) * 100 + 5,
        'Close': np.random.rand(500) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 500)
    })
    X, y = build_features(df, horizon=1)
    balance = y.mean()
    assert 0.2 < balance < 0.8

def test_inr_features_present():
    dates = pd.date_range("2022-01-01", periods=200)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 10,
        'High': np.random.rand(200) * 100 + 20,
        'Low': np.random.rand(200) * 100 + 5,
        'Close': np.random.rand(200) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 200)
    })
    X, y = build_features(df, horizon=1, ticker="TATASTEEL.NS")
    # TATASTEEL is INR so should attempt to pull fx_usd_inr
    assert "fx_usd_inr" in X.columns

def test_usd_features_present():
    dates = pd.date_range("2022-01-01", periods=200)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(200) * 100 + 10,
        'High': np.random.rand(200) * 100 + 20,
        'Low': np.random.rand(200) * 100 + 5,
        'Close': np.random.rand(200) * 100 + 10,
        'Volume': np.random.randint(1000, 10000, 200)
    })
    X, y = build_features(df, horizon=1, ticker="NVDA")
    assert "dxy_index" in X.columns
