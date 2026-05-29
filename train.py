import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

TICKER     = "AAPL"
PERIOD     = "5y"
WINDOW     = 60
TEST_SPLIT = 0.2
EPOCHS     = 50
BATCH      = 32

print(f"Downloading {TICKER} data...")
df     = yf.download(TICKER, period=PERIOD)
data   = df[["Close"]].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i, 0])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

split   = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape(-1, WINDOW, 1)
X_test  = X_test.reshape(-1, WINDOW, 1)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

print("Training...")
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=(X_test, y_test),
    verbose=1
)

predictions = scaler.inverse_transform(model.predict(X_test))
actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(np.mean((predictions - actual) ** 2))
mae  = np.mean(np.abs(predictions - actual))
print(f"\nRMSE: ${rmse:.2f}  |  MAE: ${mae:.2f}")

plt.figure(figsize=(14, 5))
plt.plot(actual,      label="Actual",    color="blue")
plt.plot(predictions, label="Predicted", color="orange")
plt.title(f"{TICKER} — Actual vs Predicted")
plt.legend()
plt.savefig("prediction_plot.png", dpi=150)
plt.show()

model.save("stock_model.keras")
np.save("scaler_params.npy", [scaler.data_min_[0], scaler.data_max_[0]])
print("Saved: stock_model.keras")