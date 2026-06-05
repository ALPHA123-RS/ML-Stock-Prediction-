import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import shap
from typing import Dict, List, Any, Tuple
from config import MODELS_DIR, STOCK_REGISTRY
from .utils import setup_logger

logger = setup_logger("model")

def get_model(model_name: str) -> Pipeline:
    if model_name == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_name == "xgb":
        clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    elif model_name == "lgbm":
        clf = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    elif model_name == "ensemble":
        clf1 = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        clf2 = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        clf3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf = VotingClassifier(estimators=[('xgb', clf1), ('lgbm', clf2), ('rf', clf3)], voting='soft')
    else:
        raise ValueError(f"Unknown model {model_name}")

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str, ticker: str) -> Pipeline:
    logger.info(f"Training {model_name} for {ticker}")
    pipeline = get_model(model_name)
    pipeline.fit(X_train, y_train)
    
    # Save model
    save_path = MODELS_DIR / f"{ticker}_{model_name}.joblib"
    joblib.dump(pipeline, save_path)
    logger.info(f"Saved {model_name} to {save_path}")
    return pipeline

def load_model(ticker: str, model_name: str) -> Pipeline:
    save_path = MODELS_DIR / f"{ticker}_{model_name}.joblib"
    if save_path.exists():
        return joblib.load(save_path)
    raise FileNotFoundError(f"Model not found at {save_path}")

def format_confidence_label(prob: float) -> str:
    prob_max = max(prob, 1 - prob)
    if prob_max < 0.55:
        return "Weak signal"
    elif prob_max < 0.65:
        return "Moderate"
    elif prob_max < 0.75:
        return "Strong"
    else:
        return "Very strong"

def predict_today(pipeline: Pipeline, X_today: pd.DataFrame, ticker: str, horizon_days: int) -> Dict[str, Any]:
    """Predicts today's direction and returns a formatted dictionary."""
    prob = pipeline.predict_proba(X_today)[0][1]
    direction = "UP" if prob >= 0.5 else "DOWN"
    confidence = prob if direction == "UP" else 1 - prob
    
    # Extract feature importance via SHAP using the underlying model and scaler
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['classifier']
    
    X_scaled = scaler.transform(X_today)
    top_features = []
    
    try:
        if isinstance(clf, (XGBClassifier, LGBMClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_scaled)
            if isinstance(shap_vals, list):
                shap_val = shap_vals[1][0]
            else:
                shap_val = shap_vals[0]
            
            feat_imp = list(zip(X_today.columns, shap_val))
            feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feat_imp[:10]
    except Exception as e:
        logger.warning(f"Could not compute SHAP values: {e}")
    
    if ticker in STOCK_REGISTRY:
        conf = STOCK_REGISTRY[ticker]
        name = conf.name
        currency = conf.currency
    else:
        name = ticker
        from .utils import get_currency_info
        currency, _ = get_currency_info(ticker)
        
    return {
        "ticker": ticker,
        "name": name,
        "currency": currency,
        "price": float(X_today['Close'].iloc[-1]) if 'Close' in X_today.columns else 0.0,
        "direction": direction,
        "confidence": float(confidence),
        "confidence_label": format_confidence_label(confidence),
        "horizon_days": horizon_days,
        "signal_date": str(X_today.index[-1].date()) if not X_today.empty else "Unknown",
        "top_features": top_features
    }
