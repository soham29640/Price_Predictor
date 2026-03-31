import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.data_loader import fetch_data


def prepare_data(series, window_size=60):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


# 🔥 Fetch stable data
print("📡 Fetching training data...")
df = fetch_data("AAPL", interval="5m", period="5d")

# ✅ Safety checks
if df is None or df.empty:
    raise ValueError("No data fetched — check API")

if 'Close' not in df.columns:
    raise ValueError("Missing 'Close' column in data")

if len(df) < 100:
    raise ValueError("Not enough data for training")

# 🔄 Scaling
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 🧠 Prepare sequences
X, y = prepare_data(scaled_close, window_size=60)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 🤖 Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss=MeanSquaredError())

print("🚀 Training model...")
model.fit(
    X, y,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 💾 Save
os.makedirs("models", exist_ok=True)

model.save("models/price_model.h5")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model + scaler saved successfully")