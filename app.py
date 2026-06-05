import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from config import STOCK_REGISTRY, DEFAULT_HORIZON
from src.data_loader import fetch_stock_data, get_company_info
from src.features import build_features
from src.model import load_model, predict_today, train_model
from src.utils import format_price
from src.backtester import run_backtest

st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Predictor Settings")
    
    live_mode = st.toggle("🔴 Live Mode (Auto-Refresh)", value=True)
    
    st.markdown("### Quick Pick")
    cols = st.columns(4)
    if cols[0].button("NVDA"): st.session_state.ticker = "NVDA"
    if cols[1].button("GOOGL"): st.session_state.ticker = "GOOGL"
    if cols[2].button("NKE"): st.session_state.ticker = "NKE"
    if cols[3].button("TATA ₹"): st.session_state.ticker = "TATASTEEL.NS"
    
    custom_ticker = st.text_input("— or type any ticker —", value=st.session_state.get("ticker", "NVDA"))
    
    horizon_str = st.radio("Prediction horizon:", ["1 day", "3 days", "5 days", "10 days"])
    horizon_map = {"1 day": 1, "3 days": 3, "5 days": 5, "10 days": 10}
    horizon = horizon_map[horizon_str]
    
    model_choice = st.selectbox("Model:", ["xgb", "lgbm", "rf", "lr", "ensemble"])
    
    run_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

if run_btn or 'ticker' not in st.session_state:
    st.session_state.ticker = custom_ticker
    st.session_state.run_analysis = True

ticker = st.session_state.get('ticker', 'NVDA').upper()

# Load Data
@st.cache_data(ttl=3600)
def load_data_cached(tkr):
    start = "2021-01-01"
    df = fetch_stock_data(tkr, start, str(datetime.today().date()))
    info = get_company_info(tkr)
    return df, info

df, info = load_data_cached(ticker)

if df.empty:
    st.error(f"Could not load data for {ticker}. Please check the ticker symbol.")
    st.stop()

# Build Features
@st.cache_data(ttl=3600)
def get_features(tkr, df_in, horiz):
    X, y = build_features(df_in, horizon=horiz, ticker=tkr)
    return X, y

X, y = get_features(ticker, df, horizon)

# Try Load Model
try:
    pipeline = load_model(ticker, model_choice)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    
# Top Header
flag = "🇮🇳" if info.get('currency') == "INR" else "🇺🇸"
st.markdown(f"## {flag} {info.get('name', ticker)} ({ticker})")
curr_sym = STOCK_REGISTRY[ticker].currency_symbol if ticker in STOCK_REGISTRY else "$"
last_price = format_price(df['Close'].iloc[-1], curr_sym, info.get('currency', 'USD'))

st.markdown(f"**Exchange:** {info.get('exchange', 'Unknown')} | **Sector:** {info.get('sector', 'Unknown')} | **Last Price:** {last_price}")
if info.get('currency') == "INR":
    st.caption("NSE trades Mon–Fri, 9:15–15:30 IST")
else:
    st.caption("US Markets trade Mon-Fri, 9:30-16:00 EST")

tabs = st.tabs(["Overview", "Prediction", "Backtest", "Model Lab", "Compare Stocks", "Live Chart"])

# --- TAB 1: Overview ---
with tabs[0]:
    st.subheader("Price & Volume")
    show_bb = st.toggle("Show Bollinger Bands", value=False)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'].rolling(20).mean(), name="SMA 20", line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'].rolling(50).mean(), name="SMA 50", line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'].rolling(200).mean(), name="SMA 200", line=dict(color='red', width=1)), row=1, col=1)
    
    if show_bb:
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        fig.add_trace(go.Scatter(x=df['Date'], y=sma + 2*std, line=dict(color='gray', dash='dash'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=sma - 2*std, line=dict(color='gray', dash='dash'), fill='tonexty', name='BB Lower'), row=1, col=1)
    
    colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    
    cols = st.columns(4)
    cols[0].metric("52W High", format_price(df['High'].tail(252).max(), curr_sym, info.get('currency', 'USD')))
    cols[1].metric("52W Low", format_price(df['Low'].tail(252).min(), curr_sym, info.get('currency', 'USD')))
    cols[2].metric("Market Cap", info.get("market_cap", "N/A"))
    
# --- TAB 2: Prediction ---
with tabs[1]:
    if not model_loaded:
        st.warning(f"Model {model_choice} not trained for {ticker}. Please run `python train.py --ticker {ticker}`.")
    else:
        res = predict_today(pipeline, X, ticker, horizon)
        direction = res['direction']
        conf = res['confidence'] * 100
        
        st.markdown(f"### Next {horizon} Days Prediction")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            color = "green" if direction == "UP" else "red"
            arrow = "▲" if direction == "UP" else "▼"
            st.markdown(f"<h1 style='color: {color}; text-align: center; font-size: 4rem;'>{arrow} {direction}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {conf:.1f}%</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{res['confidence_label']}</p>", unsafe_allow_html=True)
            
        with c2:
            st.subheader("Top Drivers (SHAP)")
            feats = [x[0] for x in res['top_features']][::-1]
            vals = [x[1] for x in res['top_features']][::-1]
            colors_bar = ['green' if v > 0 else 'red' for v in vals]
            fig_shap = go.Figure(go.Bar(
                x=vals, y=feats, orientation='h', marker_color=colors_bar
            ))
            fig_shap.update_layout(height=300, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_shap, use_container_width=True)

# --- TAB 3: Backtest ---
with tabs[2]:
    if model_loaded:
        st.subheader("Walk-Forward Backtest")
        with st.spinner("Running backtest..."):
            bt_res = run_backtest(X, y, model_choice, ticker)
            
        eq = bt_res.equity_curve
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=eq.index, y=eq['strategy_equity'], name="Strategy Equity", line=dict(color='cyan')))
        fig_bt.add_trace(go.Scatter(x=eq.index, y=eq['buy_hold_equity'], name="Buy & Hold", line=dict(color='gray', dash='dash')))
        fig_bt.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_bt, use_container_width=True)
        
        m = bt_res.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
        c2.metric("Win Rate", f"{m['win_rate']*100:.1f}%")
        c3.metric("Cumulative Return", f"{m['cumulative_return']*100:.1f}%")
        c4.metric("Max Drawdown", f"{m['max_drawdown']*100:.1f}%")
    else:
        st.warning("Train model first to see backtest.")

# --- TAB 4: Model Lab ---
with tabs[3]:
    st.write("Train models dynamically or inspect existing ones.")
    if st.button("Train / Retrain Current Model"):
        with st.spinner(f"Training {model_choice} on {len(X)} samples..."):
            train_model(X, y, model_choice, ticker)
            st.success("Trained successfully! Reload page to see updates.")

# --- TAB 5: Compare Stocks ---
with tabs[4]:
    st.subheader("Compare Multiple Stocks (Live Market Map)")
    compare_tickers = st.multiselect("Select stocks to compare:", list(STOCK_REGISTRY.keys()), default=["NVDA", "GOOGL"])
    
    if st.button("Run Comparison") or 'compare_run' not in st.session_state:
        st.session_state.compare_run = True

    if st.session_state.get('compare_run', False):
        results = []
        import yfinance as yf
        for t in compare_tickers:
            try:
                tk = yf.Ticker(t)
                last_price = tk.fast_info.last_price
                prev_close = tk.fast_info.previous_close
                pct_change = ((last_price - prev_close) / prev_close) * 100
                mcap = tk.fast_info.market_cap if hasattr(tk.fast_info, 'market_cap') else 1e9
                
                results.append({
                    "Stock": t,
                    "Last Price": last_price,
                    "Change %": pct_change,
                    "Market Cap": mcap
                })
            except Exception as e:
                pass
                
        if results:
            comp_df = pd.DataFrame(results)
            import plotly.express as px
            fig_map = px.treemap(
                comp_df,
                path=['Stock'],
                values='Market Cap',
                color='Change %',
                color_continuous_scale=['#ff4d4d', '#262626', '#00e676'],
                color_continuous_midpoint=0,
                custom_data=['Change %', 'Last Price']
            )
            fig_map.data[0].texttemplate = "<b>%{label}</b><br>%{customdata[0]:.2f}%<br>$%{customdata[1]:.2f}"
            fig_map.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=500, template="plotly_dark")
            st.plotly_chart(fig_map, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True)

# --- TAB 6: Live Chart ---
with tabs[5]:
    st.subheader(f"Real-Time Intraday Chart - {ticker}")
    auto_refresh = st.checkbox("Auto-Refresh (10s)", value=False)
    
    @st.cache_data(ttl=10)
    def get_live_data(tkr):
        import yfinance as yf
        return yf.download(tkr, period="1d", interval="1m", progress=False)

    df_live = get_live_data(ticker)
    if not df_live.empty:
        if isinstance(df_live.columns, pd.MultiIndex):
            df_live.columns = [col[0] if isinstance(col, tuple) else col for col in df_live.columns]
        df_live = df_live.reset_index()
        time_col = 'Datetime' if 'Datetime' in df_live.columns else ('index' if 'index' in df_live.columns else 'Date')
        
        fig_live = go.Figure(data=[go.Candlestick(x=df_live[time_col],
                        open=df_live['Open'],
                        high=df_live['High'],
                        low=df_live['Low'],
                        close=df_live['Close'],
                        name="Live")])
        fig_live.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=True,
                               title=f"{ticker} 1-Minute Live Action")
        st.plotly_chart(fig_live, use_container_width=True)
        
        if auto_refresh:
            st.caption("Live mode managed globally via sidebar.")
    else:
        st.info("Live data not available currently. The market might be closed or the ticker is invalid.")

if live_mode:
    import time
    time.sleep(10)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
