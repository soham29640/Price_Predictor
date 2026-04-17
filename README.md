# 📊 VolatiX: Real-Time Stock Price Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**VolatiX** is a production-ready, real-time stock price prediction dashboard powered by a multi-layer **LSTM deep learning model**. It ingests live OHLCV market data, runs iterative multi-step forecasting, and renders fully interactive Plotly charts — all inside a Streamlit web application that refreshes automatically every 3 minutes.

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Data Flow Diagram](#-data-flow-diagram)
4. [ML Pipeline — Step by Step](#-ml-pipeline--step-by-step)
5. [File Execution Walkthrough](#-file-execution-walkthrough)
6. [Model Architecture](#-model-architecture)
7. [Setup & Installation](#-setup--installation)
8. [Usage](#-usage)
9. [Dependencies](#-dependencies)
10. [Security Notes](#-security-notes)
11. [License](#-license)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📡 **Live market data** | Fetches real-time OHLCV data via Yahoo Finance (Alpha Vantage fallback → synthetic fallback) |
| 🤖 **LSTM-based forecasting** | Deep learning model trained on 5-minute interval close prices |
| 📈 **Interactive charts** | Candlestick chart + multi-step price forecast overlay via Plotly |
| ⏱️ **Auto-refresh** | Dashboard refreshes every 3 minutes; data is cached for the same window |
| 🔧 **Configurable** | Ticker symbol and prediction horizon are adjustable via the sidebar |
| 🔄 **Scheduled retraining** | Background process retrains the model from scratch every hour |

---

## 🗂️ Project Structure

```
Price_Predictor/
├── app.py                                    # Streamlit dashboard — main entry point
├── requirements.txt                          # Python dependencies
├── .env                                      # API keys (git-ignored — see setup)
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── Untitled.ipynb                        # Exploratory / experimental notebook
│
├── src/
│   ├── train/
│   │   ├── train_price_model.py              # One-shot LSTM training script
│   │   └── auto_trainer.py                   # Periodic retraining loop (every hour)
│   └── utils/
│       ├── data_loader.py                    # Data fetching with fallback chain
│       └── load_and_predict_price_model.py   # Load saved model & generate predictions
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
│                        TRAINING PIPELINE (offline)                          │
│                                                                             │
│  train_price_model.py                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  data_loader.fetch_data("AAPL", interval="5m", period="1mo")               │
│       │  Yahoo Finance ──► success?  ──Yes──► OHLCV DataFrame              │
│       │         No ▼                                                        │
│       │  Alpha Vantage ──► success?  ──Yes──► OHLCV DataFrame              │
│       │         No ▼                                                        │
│       │  Synthetic fallback ──────────────► OHLCV DataFrame                │
│       │                                          │                          │
│       ▼                                          ▼                          │
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
│  app.py  ──► st_autorefresh (every 3 min)                                  │
│       │                                                                     │
│       ▼                                                                     │
│  data_loader.fetch_data(ticker, interval="5m", period="5d")                │
│       │  Yahoo Finance ──► success?  ──Yes──► OHLCV DataFrame              │
│       │         No ▼                                                        │
│       │  Alpha Vantage ──► success?  ──Yes──► OHLCV DataFrame              │
│       │         No ▼                                                        │
│       │  Synthetic fallback ──────────────► OHLCV DataFrame                │
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
│                    CONTINUOUS RETRAINING (background)                       │
│                                                                             │
│  auto_trainer.py                                                            │
│       │                                                                     │
│       └── loop every 3600 s:                                                │
│               subprocess.run(train_price_model.py)                         │
│               ──► overwrites models/price_model.h5 + scaler.pkl            │
│               ──► app.py picks up new model on next request                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 ML Pipeline — Step by Step

### Phase 1 — Data Ingestion (`src/utils/data_loader.py`)

`fetch_data(ticker, interval, period)` implements a **three-tier fallback chain** to guarantee data availability:

| Priority | Source | Condition |
|---|---|---|
| 1 | Yahoo Finance (`yfinance`) | Primary — real-time 5-minute OHLCV bars |
| 2 | Alpha Vantage REST API | Secondary — daily OHLCV if Yahoo fails |
| 3 | Synthetic DataFrame | Hard fallback — sequential prices; ensures app never crashes |

The function always returns a normalised DataFrame with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

---

### Phase 2 — Model Training (`src/train/train_price_model.py`)

Executed **once manually** before first use, then **automatically every hour** via `auto_trainer.py`.

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

### Phase 3 — Live Inference (`src/utils/load_and_predict_price_model.py`)

Called by `app.py` on every dashboard render cycle.

```
Step 1  Load models/price_model.h5  (compile=False for speed)
        Load models/scaler.pkl

Step 2  Extract last 60 Close prices from live DataFrame
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

### Phase 4 — Dashboard Rendering (`app.py`)

```
Step 1  st_autorefresh(interval=180_000 ms)      # trigger every 3 minutes
Step 2  load_data(ticker)                        # calls fetch_data, cached 180 s
Step 3  Validate DataFrame (columns, length ≥ 60)
Step 4  Build Candlestick chart from OHLCV data
Step 5  predict_next_prices(df, window=60, horizon)
Step 6  Generate future timestamps  (last_date + 5 min × horizon)
Step 7  Build Forecast line chart (last 100 actuals + predictions)
Step 8  Compute price metrics  (current price, next predicted, % change)
Step 9  Render metric tiles + both Plotly charts to browser
```

---

### Phase 5 — Continuous Retraining (`src/train/auto_trainer.py`)

```
while True:
    subprocess.run(["python", "src/train/train_price_model.py"])
    # ↑ full train cycle: fetch → scale → train → save
    time.sleep(3600)   # wait 1 hour, then repeat
```

The model artifacts are **overwritten in place**. The dashboard loads the model fresh on each request cycle, so the updated model is picked up automatically without restarting the app.

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

### 4. Configure environment variables

Create a `.env` file in the project root (git-ignored):

```env
ALPHA_API_KEY=your_alpha_vantage_api_key_here
```

> **Note:** An Alpha Vantage key is only used as a fallback when Yahoo Finance is unavailable. A free key is available at [alphavantage.co](https://www.alphavantage.co/support/#api-key).

---

## 🚀 Usage

### Step 1 — Train the initial model

```bash
python src/train/train_price_model.py
```

Fetches 1 month of 5-minute AAPL data, trains the LSTM, and saves `models/price_model.h5` and `models/scaler.pkl`.

### Step 2 — (Optional) Start continuous retraining

Run in a separate terminal to retrain the model every hour in the background:

```bash
python src/train/auto_trainer.py
```

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

- Enter any valid stock ticker (`AAPL`, `TSLA`, `MSFT`, etc.) in the sidebar.
- Adjust the **Prediction Horizon** slider to set how many 5-minute steps ahead to forecast.
- The dashboard auto-refreshes every **3 minutes**.

---

## 📦 Dependencies

| Package | Version | Role |
|---|---|---|
| `streamlit` | 1.35.0 | Web dashboard framework |
| `tensorflow` | 2.15.0 | LSTM model training & inference |
| `yfinance` | 0.2.38 | Yahoo Finance data fetching |
| `plotly` | 5.22.0 | Interactive candlestick & forecast charts |
| `pandas` | 2.2.2 | DataFrame manipulation |
| `numpy` | 1.26.4 | Numerical operations & array shaping |
| `scikit-learn` | 1.4.2 | MinMaxScaler for data normalisation |
| `joblib` | 1.4.2 | Scaler serialisation / deserialisation |
| `arch` | 6.3.0 | Volatility modelling utilities |
| `streamlit-autorefresh` | 1.0.1 | Timed dashboard refresh |
| `dotenv` | latest | `.env` file loading |

---

## 🔒 Security Notes

- Never commit your `.env` file — it is listed in `.gitignore`.
- Never commit the `models/` directory — model artifacts are also git-ignored.
- Keep your Alpha Vantage API key private and rotate it if exposed.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
