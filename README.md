# 📊 VolatiX: Real-Time Stock Price Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**VolatiX** is a production-ready stock price prediction dashboard powered by a multi-layer **LSTM deep learning model**. It reads OHLCV market data from local CSV files (kept fresh by a scheduler), runs iterative multi-step forecasting, and renders fully interactive Plotly charts — all inside a Streamlit web application that refreshes automatically every 5 minutes.

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Data Flow Diagram](#-data-flow-diagram)
4. [ML Pipeline — Step by Step](#-ml-pipeline--step-by-step)
5. [Model Architecture](#-model-architecture)
6. [Setup & Installation](#-setup--installation)
7. [Usage](#-usage)
8. [Dependencies](#-dependencies)
9. [License](#-license)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📡 **Local CSV data** | Reads OHLCV data from `data/raw/{ticker}.csv`; refreshed hourly by the scheduler |
| 🤖 **LSTM-based forecasting** | Deep learning model trained on 5-minute interval close prices |
| 📈 **Interactive charts** | Candlestick chart + multi-step price forecast overlay via Plotly |
| ⏱️ **Auto-refresh** | Dashboard refreshes every 5 minutes; data is cached for 60 seconds |
| 🔧 **Configurable** | Ticker symbol and prediction horizon are adjustable via the sidebar |
| 🔄 **Scheduled updates** | Background scheduler updates CSV data every hour and retrains the model every 24 hours |

---

## 🗂️ Project Structure

```
Price_Predictor/
├── app.py                                    # Streamlit dashboard — main entry point
├── auto_trainer.py                           # Scheduler: data update (hourly) + retraining (daily)
├── requirements.txt                          # Python dependencies
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── Untitled.ipynb                        # Exploratory / experimental notebook
│
├── data/
│   └── raw/                                  # CSV data files (git-ignored)
│       └── {TICKER}.csv                      # e.g. AAPL.csv
│
├── src/
│   ├── data_loader.py                        # OHLCV data fetching via yfinance
│   ├── update_data.py                        # Fetches latest data and saves to CSV
│   ├── train_price_model.py                  # One-shot LSTM training script
│   └── load_and_predict_price_model.py       # Load saved model & generate predictions
│
└── models/                                   # Saved artifacts (git-ignored)
    ├── price_model.h5                        # Trained Keras LSTM model
    └── scaler.pkl                            # Fitted MinMaxScaler
```

---

## 🔄 Data Flow Diagram

The diagram below traces every data transformation from raw market feed to rendered forecast.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA UPDATE PIPELINE (src/update_data.py)              │
│                                                                             │
│  update_data.py                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  data_loader.fetch_data("AAPL", interval="5m", period="5d")                │
│       │  Yahoo Finance ──► success? ──Yes──► OHLCV DataFrame               │
│       │         No ▼                                                        │
│       │  Retry (up to 2 attempts, 60 s delay on rate limit)                │
│       │                                                                     │
│       ▼                                                                     │
│  data/raw/AAPL.csv  ◄── saved to disk                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (src/train_price_model.py)         │
│                                                                             │
│  train_price_model.py                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  data_loader.fetch_data("AAPL", interval="5m", period="1mo")               │
│       │  Yahoo Finance ──► OHLCV DataFrame                                 │
│       │                          │                                          │
│       ▼                          ▼                                          │
│  MinMaxScaler.fit_transform(Close prices)                                   │
│       │                                                                     │
│       ▼                                                                     │
│  Sliding window (size=60) ──► [X: (n,60,1)]  [y: (n,1)]                   │
│       │                                                                     │
│       ▼                                                                     │
│  LSTM model training (20 epochs, EarlyStopping patience=3)                 │
│       │                                                                     │
│       ▼                                                                     │
│  models/price_model.h5   models/scaler.pkl  ◄── Artifacts saved to disk   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE (live / app.py)                    │
│                                                                             │
│  Browser (localhost:8501)                                                   │
│       │  User enters ticker + prediction horizon                            │
│       ▼                                                                     │
│  app.py  ──► st_autorefresh (every 5 min)                                  │
│       │                                                                     │
│       ▼                                                                     │
│  load_data(ticker)                                                          │
│       │  Read data/raw/{ticker}.csv ──► OHLCV DataFrame (cached 60 s)      │
│       │                                          │                          │
│       ▼                                          │                          │
│  Candlestick chart (Plotly)  ◄───────────────────┘                         │
│                                                                             │
│       ▼                                                                     │
│  load_and_predict_price_model.predict_next_prices(df, window=60, horizon)  │
│       │                                                                     │
│       ├── Load models/price_model.h5  (Keras LSTM)                         │
│       ├── Load models/scaler.pkl      (MinMaxScaler)                       │
│       ├── Scale last 60 Close prices                                        │
│       ├── Iterative forecast loop (horizon steps):                         │
│       │       predict next step ──► append to window ──► shift window      │
│       └── Inverse-transform predictions ──► real price values              │
│                                          │                                  │
│       ▼                                  ▼                                  │
│  Forecast line chart (Plotly)   Current / Next price metrics                │
│       │                                                                     │
│       ▼                                                                     │
│  Streamlit renders both charts + metric tiles to browser                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULER (auto_trainer.py)                              │
│                                                                             │
│  auto_trainer.py                                                            │
│       │                                                                     │
│       └── loop every 60 s:                                                  │
│               if ≥ 1 hour since last update:                               │
│                   subprocess.run(src/update_data.py)                       │
│                   ──► refreshes data/raw/{ticker}.csv                      │
│               if ≥ 24 hours since last train:                              │
│                   subprocess.run(src/train_price_model.py)                 │
│                   ──► overwrites models/price_model.h5 + scaler.pkl        │
│               ──► app.py picks up new data/model on next request           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 ML Pipeline — Step by Step

### Phase 1 — Data Fetching (`src/data_loader.py`)

`fetch_data(ticker, interval, period)` fetches OHLCV data from **Yahoo Finance** via `yfinance`:

| Behaviour | Detail |
|---|---|
| Source | Yahoo Finance (`yfinance`) — 5-minute OHLCV bars |
| Rate limit handling | Retries up to 2 times with a 60-second delay |
| Column normalisation | Renames `Datetime` → `Date`, falls back to `Adj Close` if `Close` is missing |
| Output | Clean DataFrame with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume` |

---

### Phase 2 — Data Update (`src/update_data.py`)

Fetches the latest 5 days of 5-minute data for the configured ticker and writes it to `data/raw/{ticker}.csv`. Run manually before first use or let `auto_trainer.py` handle it hourly.

---

### Phase 3 — Model Training (`src/train_price_model.py`)

Executed **once manually** before first use, then **automatically every 24 hours** via `auto_trainer.py`.

```
Step 1  fetch_data("AAPL", interval="5m", period="1mo")
            │
            ▼
Step 2  MinMaxScaler.fit_transform(Close prices → [0, 1])
            │
            ▼
Step 3  Sliding-window feature engineering
            window_size = 60 bars (~5 hours of 5-min data)
            X.shape = (N, 60, 1)   y.shape = (N, 1)
            │
            ▼
Step 4  Build Sequential LSTM model
            LSTM(128) → BN → Dropout(0.3)
            LSTM(64)  → BN → Dropout(0.3)
            LSTM(32)  → Dropout(0.2)
            Dense(64, relu, L2) → Dense(32, relu) → Dense(1)
            │
            ▼
Step 5  model.fit(epochs=20, batch_size=32, val_split=0.1,
                  EarlyStopping(patience=3, restore_best_weights=True))
            │
            ▼
Step 6  Save artifacts
            models/price_model.h5   ← Keras LSTM weights
            models/scaler.pkl       ← fitted MinMaxScaler
```

---

### Phase 4 — Live Inference (`src/load_and_predict_price_model.py`)

Called by `app.py` on every dashboard render cycle.

```
Step 1  Load models/price_model.h5  (compile=False for speed)
        Load models/scaler.pkl

Step 2  Extract last 60 Close prices from DataFrame
        scaler.transform(prices)  →  scaled input (60, 1)

Step 3  Reshape to (1, 60, 1)  — batch of 1 sequence

Step 4  Iterative multi-step forecasting loop:
        for step in range(horizon):
            next_scaled = model.predict(inputs)   # shape (1,1)
            append to predictions list
            shift window: drop oldest timestep, append new prediction

Step 5  scaler.inverse_transform(predictions)  →  real price array (horizon,)

Step 6  Return predicted prices to app.py
```

---

### Phase 5 — Dashboard Rendering (`app.py`)

```
Step 1  st_autorefresh(interval=300_000 ms)      # trigger every 5 minutes
Step 2  load_data(ticker)                        # reads CSV, cached 60 s
Step 3  Validate DataFrame (columns, length ≥ 60)
Step 4  Build Candlestick chart from OHLCV data
Step 5  predict_next_prices(df, window=60, horizon)
Step 6  Generate future timestamps  (last_date + 5 min × horizon)
Step 7  Build Forecast line chart (last 100 actuals + predictions)
Step 8  Compute price metrics  (current price, next predicted, % change)
Step 9  Render metric tiles + both Plotly charts to browser
```

---

### Phase 6 — Scheduler (`auto_trainer.py`)

```
while True:
    if time_since_last_update >= 3600:
        subprocess.run(["python", "src/update_data.py"])
        # ↑ fetches latest CSV data from Yahoo Finance

    if time_since_last_train >= 86400:
        subprocess.run(["python", "src/train_price_model.py"])
        # ↑ full train cycle: fetch → scale → train → save

    time.sleep(60)   # check every minute
```

The model artifacts and CSV data are **overwritten in place**. The dashboard picks up updates automatically on the next request cycle.

---

## 🧠 Model Architecture

| Layer | Configuration |
|---|---|
| LSTM — 128 units | `return_sequences=True`, input `(60, 1)` |
| BatchNormalization | — |
| Dropout | rate = 0.3 |
| LSTM — 64 units | `return_sequences=True` |
| BatchNormalization | — |
| Dropout | rate = 0.3 |
| LSTM — 32 units | — |
| Dropout | rate = 0.2 |
| Dense — 64, ReLU | L2 regularisation (λ = 0.001) |
| Dense — 32, ReLU | — |
| Dense — 1 | Output: next scaled close price |

| Hyperparameter | Value |
|---|---|
| Window size | 60 time steps (5 hours @ 5 min bars) |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | patience = 3, restore best weights |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/soham29640/Price_Predictor.git
cd Price_Predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Fetch initial data

```bash
python src/update_data.py
```

Downloads the latest 5 days of 5-minute AAPL OHLCV data from Yahoo Finance and saves it to `data/raw/AAPL.csv`.

### Step 2 — Train the initial model

```bash
python src/train_price_model.py
```

Fetches 1 month of 5-minute AAPL data, trains the LSTM, and saves `models/price_model.h5` and `models/scaler.pkl`.

### Step 3 — (Optional) Start the scheduler

Run in a separate terminal to keep data and the model up-to-date automatically:

```bash
python auto_trainer.py
```

This updates `data/raw/{ticker}.csv` every hour and retrains the model every 24 hours.

### Step 4 — Launch the dashboard

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

- Enter any valid stock ticker (`AAPL`, `TSLA`, `MSFT`, etc.) in the sidebar.
- Adjust the **Prediction Horizon** slider to set how many 5-minute steps ahead to forecast.
- The dashboard auto-refreshes every **5 minutes**.

> **Note:** The app reads data from `data/raw/{ticker}.csv`. If the file does not exist for the ticker you enter, run `python src/update_data.py` (edit the `TICKER` variable as needed) or let the scheduler create it.

---

## 📦 Dependencies

| Package | Version | Role |
|---|---|---|
| `streamlit` | 1.35.0 | Web dashboard framework |
| `tensorflow-cpu` | 2.16.1 | LSTM model training & inference |
| `yfinance` | 0.2.54 | Yahoo Finance data fetching |
| `plotly` | 5.22.0 | Interactive candlestick & forecast charts |
| `pandas` | 2.2.2 | DataFrame manipulation |
| `numpy` | 1.26.4 | Numerical operations & array shaping |
| `scikit-learn` | 1.4.2 | MinMaxScaler for data normalisation |
| `joblib` | 1.4.2 | Scaler serialisation / deserialisation |
| `streamlit-autorefresh` | 1.0.1 | Timed dashboard refresh |

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
