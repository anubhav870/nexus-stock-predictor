import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────
TICKER   = "AAPL"      # Change to any ticker: MSFT, TSLA, RELIANCE.NS
PERIOD   = "5y"        # 5 years of data
WINDOW   = 60          # Look back 60 days to predict next day
TEST_SPLIT = 0.2       # 20% held out for testing
EPOCHS   = 50
BATCH    = 32
# ─────────────────────────────────────────────────────────

# 1. Download data
print(f"Downloading {TICKER} data...")
df = yf.download(TICKER, period=PERIOD)
data = df[["Close"]].values          # Use closing price only

# 2. Scale to [0, 1]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 3. Build sequences: X = 60 days, y = next day
X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

# 4. Train/test split (keep chronological order!)
split = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(-1, WINDOW, 1)
X_test  = X_test.reshape(-1, WINDOW, 1)

# 5. Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# 6. Train
print("Training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. Evaluate
predictions = scaler.inverse_transform(model.predict(X_test))
actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(np.mean((predictions - actual) ** 2))
mae  = np.mean(np.abs(predictions - actual))
print(f"\nRMSE: ${rmse:.2f}  |  MAE: ${mae:.2f}")

# 8. Plot
plt.figure(figsize=(14, 5))
plt.plot(actual,      label="Actual",    color="blue")
plt.plot(predictions, label="Predicted", color="orange")
plt.title(f"{TICKER} — Actual vs Predicted (Test Set)")
plt.xlabel("Days"); plt.ylabel("Price (USD)")
plt.legend(); plt.tight_layout()
plt.savefig("prediction_plot.png", dpi=150)
plt.show()

# 9. Save model + scaler params
model.save("stock_model.keras")
np.save("scaler_params.npy", [scaler.data_min_[0], scaler.data_max_[0]])
print("Saved: stock_model.keras")