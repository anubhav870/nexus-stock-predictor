import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.graph_objects as go
import datetime

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("stock_model.keras")

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    info = {}
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        pass
    return df, info

st.set_page_config(
    page_title="NEXUS · Stock AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --gold:   #c9a84c;
    --gold2:  #f0d080;
    --bg:     #080810;
    --bg2:    #0d0d1c;
    --bg3:    #12122a;
    --border: #1c1c3a;
    --bord2:  #2a2a50;
    --text:   #e8e4d8;
    --muted:  #5a5a80;
    --green:  #00e5a0;
    --red:    #ff4566;
    --blue:   #3d9fff;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: var(--bg) !important; color: var(--text); }
.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: linear-gradient(rgba(201,168,76,0.025) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(201,168,76,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none; z-index: 0;
}

section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.nexus-logo { padding: 28px 20px 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.nexus-wordmark { font-family: 'Bebas Neue', sans-serif; font-size: 32px; letter-spacing: 6px; background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 50%, var(--gold) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }
.nexus-sub { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 4px; color: var(--muted); margin-top: 4px; }

.sb-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--gold); margin: 20px 0 8px; opacity: 0.8; }

.stButton > button {
    background: transparent !important; border: 1px solid var(--bord2) !important; border-radius: 6px !important;
    color: var(--muted) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
    padding: 6px 4px !important; transition: all 0.2s !important; width: 100% !important; letter-spacing: 1px !important;
}
.stButton > button:hover { border-color: var(--gold) !important; color: var(--gold) !important; background: rgba(201,168,76,0.06) !important; }

.predict-wrap .stButton > button {
    background: linear-gradient(135deg, #b8860b 0%, var(--gold) 50%, var(--gold2) 100%) !important;
    color: #000 !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 18px !important;
    letter-spacing: 4px !important; padding: 14px !important; border: none !important; border-radius: 8px !important;
    box-shadow: 0 0 30px rgba(201,168,76,0.25), 0 4px 20px rgba(0,0,0,0.5) !important;
}
.predict-wrap .stButton > button:hover { box-shadow: 0 0 50px rgba(201,168,76,0.45), 0 4px 20px rgba(0,0,0,0.5) !important; transform: translateY(-1px) !important; }

.stTextInput > div > div > input { background: var(--bg3) !important; border: 1px solid var(--bord2) !important; border-radius: 8px !important; color: var(--gold2) !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 22px !important; letter-spacing: 4px !important; text-align: center !important; }
.stTextInput > div > div > input:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important; }

.stSelectbox > div > div { background: var(--bg3) !important; border: 1px solid var(--bord2) !important; border-radius: 8px !important; color: var(--text) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }

.stSlider > div > div > div > div { background: linear-gradient(90deg, var(--gold), var(--gold2)) !important; }

.main-header { padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 32px; display: flex; align-items: flex-end; justify-content: space-between; }
.main-title { font-family: 'Bebas Neue', sans-serif; font-size: 72px; letter-spacing: 8px; background: linear-gradient(135deg, #fff 0%, var(--gold2) 60%, var(--gold) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 0.9; }
.main-company { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 3px; color: var(--muted); margin-top: 8px; text-transform: uppercase; }

.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
.mcard { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px 22px; position: relative; overflow: hidden; transition: border-color 0.3s; }
.mcard:hover { border-color: var(--bord2); }
.mcard::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, var(--gold), transparent); opacity: 0.4; }
.mcard-icon { font-size: 18px; margin-bottom: 10px; display: block; }
.mcard-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.mcard-value { font-family: 'Bebas Neue', sans-serif; font-size: 38px; letter-spacing: 2px; color: var(--text); line-height: 1; }
.mcard-delta { font-family: 'JetBrains Mono', monospace; font-size: 12px; margin-top: 6px; font-weight: 500; }
.up   { color: var(--green) !important; }
.down { color: var(--red)   !important; }
.flat { color: var(--muted) !important; }

.sec-title { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: var(--gold); margin: 28px 0 14px; display: flex; align-items: center; gap: 12px; }
.sec-title::before { content: '◈'; font-size: 12px; }
.sec-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--bord2), transparent); }

.ftable { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.frow { display: grid; grid-template-columns: 60px 1fr 130px 110px; align-items: center; padding: 13px 20px; border-bottom: 1px solid var(--border); transition: background 0.2s; }
.frow:last-child { border-bottom: none; }
.frow:hover { background: rgba(201,168,76,0.04); }
.frow-head { background: var(--bg3); font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); }
.frow-day { font-family: 'Bebas Neue', sans-serif; font-size: 22px; color: var(--muted); letter-spacing: 1px; }
.frow-date { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }
.frow-price { font-family: 'Bebas Neue', sans-serif; font-size: 22px; letter-spacing: 1px; color: var(--text); text-align: right; }
.frow-chg { font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; }

.landing { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 55vh; gap: 20px; text-align: center; }

div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)

WINDOW = 60
QUICK_TICKERS = {
    "AAPL": "🍎", "MSFT": "🪟", "TSLA": "⚡", "NVDA": "🟩",
    "GOOGL": "🔍", "AMZN": "📦", "META": "🔵", "RELIANCE.NS": "🇮🇳",
    "TCS.NS": "💻", "INFY.NS": "🔷",
}

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='nexus-logo'>
        <div class='nexus-wordmark'>NEXUS</div>
        <div class='nexus-sub'>AI · STOCK · INTELLIGENCE</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-label'>Quick Select</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    selected_quick = None
    for i, (sym, icon) in enumerate(QUICK_TICKERS.items()):
        short = sym.replace(".NS", "")
        if cols[i % 2].button(f"{icon} {short}", key=f"q_{sym}"):
            selected_quick = sym

    st.markdown("<div class='sb-label'>Ticker Symbol</div>", unsafe_allow_html=True)
    default_ticker = selected_quick if selected_quick else st.session_state.get("last_ticker", "AAPL")
    ticker = st.text_input("", value=default_ticker, label_visibility="collapsed").upper().strip()
    st.session_state["last_ticker"] = ticker

    st.markdown("<div class='sb-label'>Historical Period</div>", unsafe_allow_html=True)
    period = st.selectbox("", ["1y", "2y", "5y"], index=1, label_visibility="collapsed")

    st.markdown("<div class='sb-label'>Forecast Horizon</div>", unsafe_allow_html=True)
    n_days = st.slider("", 1, 30, 7, label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='predict-wrap'>", unsafe_allow_html=True)
    predict_btn = st.button("PREDICT", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:32px;padding-top:20px;border-top:1px solid #1c1c3a'>
        <div style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:2px;color:#2a2a50;line-height:2'>
            POWERED BY LSTM NEURAL NET<br>DATA VIA YAHOO FINANCE<br>FOR EDUCATIONAL USE ONLY
        </div>
    </div>""", unsafe_allow_html=True)

# ── LANDING ───────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div class='landing'>
        <div style='font-family:Bebas Neue,sans-serif;font-size:120px;letter-spacing:12px;
                    background:linear-gradient(135deg,#1c1c3a 0%,#2a2a50 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;line-height:1'>◈</div>
        <div style='font-family:Bebas Neue,sans-serif;font-size:48px;letter-spacing:8px;
                    background:linear-gradient(135deg,#fff 0%,#c9a84c 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text'>STOCK INTELLIGENCE</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:11px;letter-spacing:4px;color:#5a5a80;text-transform:uppercase'>
            Select a ticker · Set horizon · Click predict
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── PREDICTION ────────────────────────────────────────
with st.spinner(""):
    try:
        model = get_model()
    except Exception:
        st.error("No trained model found. Run `python train.py` first.")
        st.stop()

    df, info = get_stock_data(ticker, period)
    if df.empty:
        st.error(f"No data found for **{ticker}**.")
        st.stop()

    closes = df["Close"].values.reshape(-1, 1)
    closes = closes.astype(float)
    dates  = df.index

    scaler = MinMaxScaler()
    scaler.fit(closes)
    scaled = scaler.transform(closes)

    X_hist = []
    for i in range(WINDOW, len(scaled)):
        X_hist.append(scaled[i - WINDOW:i, 0])
    X_hist = np.array(X_hist).reshape(-1, WINDOW, 1)

    hist_preds  = scaler.inverse_transform(model.predict(X_hist, verbose=0))
    hist_actual = closes[WINDOW:]
    hist_dates  = dates[WINDOW:]

    rmse = float(np.sqrt(np.mean((hist_preds.flatten() - hist_actual.flatten()) ** 2)))
    mae  = float(np.mean(np.abs(hist_preds.flatten() - hist_actual.flatten())))

    last_seq  = scaled[-WINDOW:].reshape(1, WINDOW, 1)
    forecasts = []
    seq       = last_seq.copy()
    for _ in range(n_days):
        pred = model.predict(seq, verbose=0)[0, 0]
        forecasts.append(pred)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = pred

    forecast_prices = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    future_dates    = pd.date_range(start=df.index[-1], periods=n_days + 1)[1:]
    current_price   = float(closes[-1].flatten()[0])
    forecast_end    = float(forecast_prices[-1])
    pct_change      = (forecast_end / current_price - 1) * 100

# ── HEADER ────────────────────────────────────────────
company_name = info.get("shortName", ticker)
direction    = "▲" if pct_change >= 0 else "▼"
dir_color    = "#00e5a0" if pct_change >= 0 else "#ff4566"

st.markdown(f"""
<div class='main-header'>
    <div>
        <div class='main-title'>{ticker}</div>
        <div class='main-company'>{company_name} &nbsp;·&nbsp; {period} period &nbsp;·&nbsp; {n_days}-day forecast</div>
    </div>
    <div style='text-align:right'>
        <div style='font-family:Bebas Neue,sans-serif;font-size:52px;letter-spacing:4px;color:{dir_color};line-height:1'>
            {direction} {abs(pct_change):.2f}%
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:2px;color:#5a5a80;margin-top:4px'>
            FORECASTED CHANGE
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ── METRIC CARDS ──────────────────────────────────────
delta_cls = "up" if pct_change >= 0 else "down"
rmse_cls  = "up" if rmse < 5 else ("flat" if rmse < 15 else "down")

st.markdown(f"""
<div class='metrics-row'>
    <div class='mcard'>
        <span class='mcard-icon'>💵</span>
        <div class='mcard-label'>Current Price</div>
        <div class='mcard-value'>${current_price:.2f}</div>
        <div class='mcard-delta flat'>live market data</div>
    </div>
    <div class='mcard'>
        <span class='mcard-icon'>🎯</span>
        <div class='mcard-label'>{n_days}-Day Target</div>
        <div class='mcard-value'>${forecast_end:.2f}</div>
        <div class='mcard-delta {delta_cls}'>{direction} {abs(pct_change):.2f}% projected</div>
    </div>
    <div class='mcard'>
        <span class='mcard-icon'>📐</span>
        <div class='mcard-label'>Model RMSE</div>
        <div class='mcard-value'>${rmse:.2f}</div>
        <div class='mcard-delta {rmse_cls}'>avg price error</div>
    </div>
    <div class='mcard'>
        <span class='mcard-icon'>📏</span>
        <div class='mcard-label'>Model MAE</div>
        <div class='mcard-value'>${mae:.2f}</div>
        <div class='mcard-delta flat'>mean abs error</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── CHART 1: Historical ───────────────────────────────
st.markdown("<div class='sec-title'>Historical vs Predicted</div>", unsafe_allow_html=True)
show_n = st.slider("History window (days)", 30, min(len(hist_actual), 500), 150, key="hs")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_dates[-show_n:], y=hist_actual.flatten()[-show_n:],
    name="Actual", line=dict(color="#3d9fff", width=2),
    fill="tozeroy", fillcolor="rgba(61,159,255,0.04)",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: <b>$%{y:.2f}</b><extra></extra>"
))
fig.add_trace(go.Scatter(
    x=hist_dates[-show_n:], y=hist_preds.flatten()[-show_n:],
    name="AI Predicted", line=dict(color="#c9a84c", width=2, dash="dot"),
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: <b>$%{y:.2f}</b><extra></extra>"
))
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a14",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#5a5a80"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0a0c0", size=11), orientation="h", y=1.06, x=0),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#12122a", bordercolor="#2a2a50", font=dict(family="JetBrains Mono", size=12, color="#e8e4d8")),
    xaxis=dict(gridcolor="#12122a", showline=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#12122a", showline=False, zeroline=False, tickprefix="$", tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=30, b=0), height=380,
)
st.plotly_chart(fig, use_container_width=True)

# ── CHART 2: Forecast ─────────────────────────────────
st.markdown("<div class='sec-title'>Price Forecast</div>", unsafe_allow_html=True)

ext_dates  = pd.DatetimeIndex([df.index[-1]]).append(future_dates)
ext_prices = np.concatenate([[current_price], forecast_prices])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=list(ext_dates) + list(ext_dates[::-1]),
    y=list(ext_prices + rmse * 1.5) + list((ext_prices - rmse * 1.5)[::-1]),
    fill="toself", fillcolor="rgba(201,168,76,0.07)",
    line=dict(color="rgba(0,0,0,0)"), name="Confidence Band", hoverinfo="skip"
))
fig2.add_trace(go.Scatter(
    x=list(ext_dates) + list(ext_dates[::-1]),
    y=list(ext_prices + rmse * 0.7) + list((ext_prices - rmse * 0.7)[::-1]),
    fill="toself", fillcolor="rgba(201,168,76,0.12)",
    line=dict(color="rgba(0,0,0,0)"), name="Inner Band", hoverinfo="skip"
))
fig2.add_trace(go.Scatter(
    x=ext_dates, y=ext_prices, name="Forecast", mode="lines+markers",
    line=dict(color="#c9a84c", width=3),
    marker=dict(size=[6]*(len(ext_dates)-1)+[14], color=["#c9a84c"]*(len(ext_dates)-1)+["#f0d080"], line=dict(color="#080810", width=2)),
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: <b>$%{y:.2f}</b><extra></extra>"
))
fig2.add_vline(
    x=df.index[-1].timestamp()*1000, line=dict(color="#2a2a50", width=1, dash="dash"),
    annotation_text="TODAY", annotation_font=dict(color="#3a3a60", size=9, family="JetBrains Mono"),
    annotation_position="top left"
)
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a14",
    font=dict(family="JetBrains Mono, monospace", size=11, color="#5a5a80"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0a0c0", size=11), orientation="h", y=1.06, x=0),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#12122a", bordercolor="#2a2a50", font=dict(family="JetBrains Mono", size=12, color="#e8e4d8")),
    xaxis=dict(gridcolor="#12122a", showline=False, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#12122a", showline=False, zeroline=False, tickprefix="$", tickfont=dict(size=10)),
    margin=dict(l=0, r=0, t=30, b=0), height=340,
)
st.plotly_chart(fig2, use_container_width=True)

# ── FORECAST TABLE ────────────────────────────────────
st.markdown("<div class='sec-title'>Day-by-Day Breakdown</div>", unsafe_allow_html=True)

rows = ""
for i, (d, p) in enumerate(zip(future_dates, forecast_prices)):
    chg     = p - current_price
    chg_pct = (p / current_price - 1) * 100
    cls     = "up" if chg >= 0 else "down"
    arrow   = "▲" if chg >= 0 else "▼"
    bar_w   = min(abs(chg_pct) * 6, 100)
    bar_col = "#00e5a0" if chg >= 0 else "#ff4566"
    rows += f"""
    <div class='frow'>
        <div class='frow-day'>{i+1:02d}</div>
        <div>
            <div class='frow-date'>{d.strftime('%A, %b %d')}</div>
            <div style='margin-top:5px;height:2px;width:{bar_w}%;background:{bar_col};border-radius:2px;opacity:0.5'></div>
        </div>
        <div class='frow-price'>${p:.2f}</div>
        <div class='frow-chg {cls}'>{arrow} {abs(chg_pct):.2f}%</div>
    </div>"""

st.markdown(f"""
<div class='ftable'>
    <div class='frow frow-head'>
        <div>DAY</div><div>DATE</div>
        <div style='text-align:right'>PRICE</div><div style='text-align:right'>CHANGE</div>
    </div>{rows}
</div>""", unsafe_allow_html=True)

# ── VOLUME ────────────────────────────────────────────
if "Volume" in df.columns:
    with st.expander("◈  VOLUME HISTORY · LAST 120 DAYS"):
        vol_colors = [
            "#00e5a0" if float(np.squeeze(c)) >= float(np.squeeze(o)) else "#ff4566"
            for c, o in zip(df["Close"].values[-120:], df["Open"].values[-120:])
        ]
        fig3 = go.Figure(go.Bar(
            x=df.index[-120:], y=df["Volume"].values[-120:],
            marker_color=vol_colors, marker_line_width=0,
            hovertemplate="<b>%{x|%b %d}</b><br>Volume: %{y:,.0f}<extra></extra>"
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a14",
            font=dict(family="JetBrains Mono, monospace", size=10, color="#5a5a80"),
            xaxis=dict(gridcolor="#12122a", showline=False, zeroline=False),
            yaxis=dict(gridcolor="#12122a", showline=False, zeroline=False),
            margin=dict(l=0, r=0, t=10, b=0), height=200, showlegend=False, bargap=0.15,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────
st.markdown("""
<div style='margin-top:48px;padding-top:20px;border-top:1px solid #1c1c3a;
            display:flex;justify-content:space-between;align-items:center'>
    <div style='font-family:Bebas Neue,sans-serif;font-size:18px;letter-spacing:4px;color:#1c1c3a'>
        NEXUS · AI STOCK INTELLIGENCE
    </div>
    <div style='font-family:JetBrains Mono,monospace;font-size:9px;letter-spacing:2px;color:#1c1c3a'>
        NOT FINANCIAL ADVICE · EDUCATIONAL USE ONLY
    </div>
</div>""", unsafe_allow_html=True)