import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from src.load_and_predict_price_model import predict_next_prices

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VolatiX Dashboard", layout="wide")
st.title("📊 VolatiX: Price Prediction Dashboard")
st.caption("Data source: Local CSV (updated hourly)")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300_000, key="auto_refresh")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
ticker  = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
horizon = st.sidebar.slider("Prediction Horizon (bars)", 5, 30, 10)
st.sidebar.caption("Each bar = 5 minutes")

WINDOW = 60


# ── Load data from CSV (NO API) ────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker: str):
    path = f"data/raw/{ticker}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run update_data.py first.")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ── Main ───────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading data from local CSV..."):
        df = load_data(ticker)

    # Basic validation
    if df.empty:
        st.error("No data available.")
        st.stop()

    required_cols = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    if len(df) < WINDOW:
        st.error(f"Need at least {WINDOW} rows, got {len(df)}.")
        st.stop()

    # Use recent data only (faster + better)
    df = df.tail(500)

    # ── Metrics ────────────────────────────────────────────────────────────────
    current_price = float(df["Close"].iloc[-1])

    with st.spinner("Running prediction model..."):
        predictions = predict_next_prices(
            df,
            window_size=WINDOW,
            horizon=horizon
        )

    next_price = float(predictions[0])
    change_pct = (next_price - current_price) / current_price * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Next Predicted Price", f"${next_price:.2f}", f"{change_pct:+.2f}%")
    col3.metric("Bars Ahead", f"{horizon} ({horizon * 5} min)")

    # ── Candlestick chart ──────────────────────────────────────────────────────
    candle_fig = go.Figure(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    candle_fig.update_layout(
        title=f"{ticker} — Recent Candlestick Data",
        xaxis_rangeslider_visible=False
    )

    # ── Forecast chart ─────────────────────────────────────────────────────────
    future_times = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(minutes=5),
        periods=horizon,
        freq="5min"
    )

    forecast = pd.Series(predictions, index=future_times)

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=df["Date"].tail(100),
        y=df["Close"].tail(100),
        mode="lines",
        name="Actual"
    ))

    forecast_fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Predicted",
        line=dict(dash="dot")
    ))

    forecast_fig.update_layout(
        title=f"{ticker} — Forecast ({horizon * 5} min ahead)",
        hovermode="x unified"
    )

    # ── Render ─────────────────────────────────────────────────────────────────
    st.plotly_chart(candle_fig, use_container_width=True)
    st.plotly_chart(forecast_fig, use_container_width=True)

    with st.expander("📋 View raw data"):
        st.dataframe(df.tail(50), use_container_width=True)

# ── Error handling ─────────────────────────────────────────────────────────────
except FileNotFoundError as e:
    st.warning(str(e))
    st.info("Run: python update_data.py")

except Exception as e:
    st.error(f"❌ {e}")