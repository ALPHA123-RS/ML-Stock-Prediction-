# Stock Predictor Pro

A complete, production-grade stock market prediction system supporting multiple exchanges.

## Architecture
```text
  [ Yahoo Finance ]
         │
         ▼ (data_loader.py)
  [ Data Validator & Cache ]
         │
         ▼ (features.py)
  [ Feature Engineering ] > 50 technical & macro features
         │
         ▼ (model.py)
  [ Sklearn ML Pipelines ] ───▶ [ XGB, LGBM, RF, LR, Ensemble ]
         │
         ▼ (backtester.py)
  [ Walk-Forward Tester ]
         │
  [ Streamlit App / CLI ]
```

## Supported Stocks

| Ticker | Name | Exchange | Currency | Benchmark |
|--------|------|----------|----------|-----------|
| NVDA | NVIDIA Corporation | NASDAQ | USD | ^GSPC |
| GOOGL | Alphabet Inc. | NASDAQ | USD | ^GSPC |
| NKE | Nike Inc. | NYSE | USD | ^GSPC |
| TATASTEEL.NS | Tata Steel Ltd | NSE | INR | ^NSEI |

## Quick-Start: Run It Now

1. **Clone & setup:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models:**
   ```bash
   python train.py --all
   ```

3. **Run App:**
   ```bash
   streamlit run app.py
   ```

## Setup Instructions
Use `environment.yml` for conda:
`conda env create -f environment.yml`
`conda activate stock_predictor`

Or pip:
`pip install -r requirements.txt`

## Adding a New Stock
Edit `config.py` and add to `STOCK_REGISTRY` dict with the exact ticker, exchange, and currency details.

## Known Limitations
- No live tick data. Predictions use end-of-day data.
- Regime shifts might not be detected immediately.
- Currency risk is captured partially via USD/INR features, but macro shocks are unpredictable.

**Disclaimer:** This software is for educational purposes only. NOT financial advice.
