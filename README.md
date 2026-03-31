# 📊 VolatiX: Real-Time Stock Price Predictor

VolatiX is a real-time stock price prediction dashboard built with **Streamlit**, **TensorFlow/Keras (LSTM)**, and **Plotly**. It fetches live market data, runs an AI-powered forecasting model, and displays interactive candlestick charts alongside future price predictions.

---

## ✨ Features

- 📡 **Live market data** — fetches real-time OHLCV data via Yahoo Finance (with Alpha Vantage as fallback)
- 🤖 **LSTM-based price prediction** — deep learning model trained on 5-minute interval close prices
- 📈 **Interactive charts** — candlestick chart + price forecast overlay using Plotly
- ⏱️ **Auto-refresh** — dashboard auto-refreshes every 3 minutes to stay current
- 🔧 **Configurable** — choose any stock ticker and prediction horizon via the sidebar
- 🔄 **Scheduled retraining** — auto-trainer retrains the model every hour in the background

---

## 🗂️ Project Structure

```
Price_Predictor/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed — see setup below)
├── .gitignore
├── LICENSE
├── notebooks/
│   └── Untitled.ipynb              # Exploratory / experimental notebook
├── src/
│   ├── train/
│   │   ├── train_price_model.py    # One-shot LSTM training script
│   │   └── auto_trainer.py         # Periodic retraining loop (every hour)
│   └── utils/
│       ├── data_loader.py          # Data fetching (Yahoo Finance → Alpha Vantage → fallback)
│       └── load_and_predict_price_model.py  # Load saved model & generate predictions
└── models/                         # Saved model artifacts (git-ignored)
    ├── price_model.h5              # Trained Keras LSTM model
    └── scaler.pkl                  # Fitted MinMaxScaler
```

---

## ⚙️ Setup

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

Create a `.env` file in the project root (this file is git-ignored):

```env
ALPHA_API_KEY=your_alpha_vantage_api_key_here
```

> **Note:** An Alpha Vantage API key is only required as a fallback when Yahoo Finance is unavailable. A free key can be obtained at [alphavantage.co](https://www.alphavantage.co/support/#api-key).

---

## 🚀 Usage

### Train the model

Before launching the dashboard, train the LSTM model:

```bash
python src/train/train_price_model.py
```

This fetches 1 month of 5-minute AAPL data, trains the model, and saves the artifacts to `models/`.

### (Optional) Start the auto-trainer

To continuously retrain the model every hour in the background:

```bash
python src/train/auto_trainer.py
```

### Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

- Enter any valid stock ticker (e.g. `AAPL`, `TSLA`, `MSFT`) in the sidebar.
- Adjust the **Prediction Horizon** slider to control how many 5-minute steps ahead to forecast.

---

## 🧠 Model Architecture

The LSTM model is a multi-layer sequential network:

| Layer | Details |
|---|---|
| LSTM (128 units) | `return_sequences=True`, input shape `(60, 1)` |
| BatchNormalization | — |
| Dropout (0.3) | — |
| LSTM (64 units) | `return_sequences=True` |
| BatchNormalization | — |
| Dropout (0.3) | — |
| LSTM (32 units) | — |
| Dropout (0.2) | — |
| Dense (64, ReLU) | L2 regularization (0.001) |
| Dense (32, ReLU) | — |
| Dense (1) | Output — next close price (scaled) |

- **Window size:** 60 time steps (5 hours of 5-minute bars)
- **Optimizer:** Adam
- **Loss:** Mean Squared Error
- **Early stopping:** patience = 3, restores best weights

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.35.0 |
| tensorflow | 2.15.0 |
| yfinance | 0.2.38 |
| plotly | 5.22.0 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| scikit-learn | 1.4.2 |
| joblib | 1.4.2 |
| arch | 6.3.0 |
| streamlit-autorefresh | 1.0.1 |

---

## 📝 File Descriptions

| File | Description |
|---|---|
| `app.py` | Streamlit application — renders the dashboard, loads data, runs predictions, and displays charts and metrics |
| `src/utils/data_loader.py` | Fetches OHLCV data; primary source is Yahoo Finance, falls back to Alpha Vantage API, then synthetic data |
| `src/utils/load_and_predict_price_model.py` | Loads the saved model and scaler, performs iterative multi-step forecasting |
| `src/train/train_price_model.py` | Fetches training data, builds and trains the LSTM model, saves `models/price_model.h5` and `models/scaler.pkl` |
| `src/train/auto_trainer.py` | Runs `train_price_model.py` in a subprocess every hour for continuous model refresh |
| `notebooks/Untitled.ipynb` | Jupyter notebook for data exploration and experimentation |

---

## 🔒 Security Notes

- Never commit your `.env` file or `models/` directory — both are listed in `.gitignore`.
- Keep your Alpha Vantage API key private.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
