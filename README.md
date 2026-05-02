# ◈ NEXUS — AI Stock Price Predictor

> Deep learning-powered stock forecasting with real-time data and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is this?

NEXUS is an end-to-end AI stock price prediction system that uses a stacked LSTM
(Long Short-Term Memory) neural network to forecast future stock prices based on
historical market data. It comes with a fully interactive web dashboard where you
can look up any stock, set a forecast horizon, and visualize predictions in real time.

---

## Features

- **LSTM Neural Network** — stacked 128→64 unit architecture with Dropout regularization
- **Real-Time Data** — live OHLCV data fetched via Yahoo Finance API (yfinance)
- **Multi-Day Forecasting** — predict 1 to 30 days ahead with confidence bands
- **Interactive Dashboard** — built with Streamlit + Plotly, fully dark-themed
- **Global Stock Support** — works with any ticker: US stocks, Indian NSE/BSE stocks, and more
- **Model Metrics** — RMSE and MAE displayed live so you always know how accurate the model is
- **Volume Analysis** — color-coded volume history chart included

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Data | yfinance (Yahoo Finance API) |
| Preprocessing | Scikit-learn, NumPy, Pandas |
| Visualization | Plotly |
| Dashboard | Streamlit |

---

## Project Structure
nexus-stock-predictor/
├── train.py        # Data pipeline, LSTM model training, model export
├── app.py          # Streamlit dashboard with live predictions
├── .gitignore      # Excludes venv, model files, cache
└── README.md       # You are here

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nexus-stock-predictor.git
cd nexus-stock-predictor
```

### 2. Create and activate a virtual environment
```bash
python -m venv stock_env

# Windows
stock_env\Scripts\activate

# Mac / Linux
source stock_env/bin/activate
```

### 3. Install dependencies
```bash
pip install yfinance numpy pandas scikit-learn tensorflow matplotlib streamlit plotly
```

### 4. Train the model
```bash
python train.py
```
This downloads 5 years of AAPL data, trains the LSTM for 50 epochs,
and saves `stock_model.keras`. Takes 5–15 minutes on CPU.

### 5. Launch the dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## How It Works
Yahoo Finance API
↓
Raw OHLCV Data
↓
MinMax Scaling + 60-day sliding window
↓
LSTM(128) → Dropout → LSTM(64) → Dense(1)
↓
Predicted Prices
↓
Streamlit Dashboard

The model looks at the past **60 trading days** to predict the next day's
closing price. For multi-day forecasts, each predicted price is fed back
into the sequence to generate the next prediction.

---

## Model Performance

Trained on AAPL (5 years of data):

| Metric | Value |
|---|---|
| RMSE | ~$4.71 |
| MAE | ~$3.64 |
| Test Split | 20% (chronological) |
| Epochs | 50 |

---

## Supported Tickers

Works with any valid Yahoo Finance ticker symbol:

| Stock | Ticker |
|---|---|
| Apple | `AAPL` |
| Microsoft | `MSFT` |
| Tesla | `TSLA` |
| Nvidia | `NVDA` |
| Reliance Industries | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| Infosys | `INFY.NS` |
| Any other stock | `TICKER` or `TICKER.NS` |

---

## Disclaimer

> This project is built for **educational purposes only**.
> It is not financial advice and should not be used to make real investment decisions.
> Stock markets are influenced by countless unpredictable factors that no model can fully capture.

---

## Author

Built by **[Anubhav Gupta]**
[LinkedIn](www.linkedin.com/in/anubhavgupta870) · [GitHub](https://github.com/anubhav870)