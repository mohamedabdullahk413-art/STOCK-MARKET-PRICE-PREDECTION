import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="Multi-Stock Price Prediction (ML)",
    layout="wide"
)

st.title("📈 Multi-Stock Price Prediction (Classical ML)")

# -------------------- TOP 20 COMPANIES --------------------
TOP_20_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Visa (V)": "V",
    "Walmart (WMT)": "WMT",
    "Procter & Gamble (PG)": "PG",
    "UnitedHealth (UNH)": "UNH",
    "Home Depot (HD)": "HD",
    "Mastercard (MA)": "MA",
    "Exxon Mobil (XOM)": "XOM",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "Netflix (NFLX)": "NFLX"
}

# -------------------- SIDEBAR --------------------
st.sidebar.header("Stock Selection")

company_name = st.sidebar.selectbox(
    "Choose a Company",
    list(TOP_20_STOCKS.keys())
)

ticker = TOP_20_STOCKS[company_name]
st.sidebar.markdown(f"**Ticker:** `{ticker}`")

lookback = st.sidebar.slider(
    "Lookback Window (Days)",
    min_value=20,
    max_value=120,
    value=60,
    step=5
)

n_estimators = st.sidebar.slider(
    "Random Forest Trees",
    min_value=100,
    max_value=500,
    value=300,
    step=50
)

# -------------------- LOAD DATA --------------------
start_date = "2023-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

st.subheader(f"📊 {company_name} Historical Data")

df = yf.download(ticker, start=start_date, end=end_date)
df = df[["Close"]].dropna()

if len(df) < lookback + 20:
    st.error("Not enough data available for this stock.")
    st.stop()

st.dataframe(df.tail())

# -------------------- PREPROCESSING --------------------
data = df[["Close"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - lookback:]

def create_lag_features(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_lag_features(train_data, lookback)
X_test, y_test = create_lag_features(test_data, lookback)

# -------------------- MODEL --------------------
@st.cache_resource
def train_model(X_train, y_train, n_estimators):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

with st.spinner("Training Random Forest model... 🌳"):
    model = train_model(X_train, y_train, n_estimators)

# -------------------- PREDICTION --------------------
predicted_scaled = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------- METRICS --------------------
mae = mean_absolute_error(actual_prices, predicted_prices)
st.metric("Mean Absolute Error (Test Set)", f"${mae:.2f}")

# -------------------- ACTUAL VS PREDICTED CHART --------------------
st.subheader("📉 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label="Actual Price", color="blue")
ax.plot(predicted_prices, label="Predicted Price", color="red")
ax.set_title(f"{ticker} Closing Price Prediction")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# -------------------- FUTURE FORECAST --------------------
st.subheader("🔮 Predicting Next 5 Business Days")

last_window = scaled_data[-lookback:].flatten()
future_predictions = []

for _ in range(5):
    pred = model.predict(last_window.reshape(1, -1))[0]
    future_predictions.append(pred)
    last_window = np.append(last_window[1:], pred)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

future_dates = pd.bdate_range(
    start=df.index[-1] + timedelta(days=1),
    periods=5
)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": future_predictions.flatten()
})

st.dataframe(future_df)

# -------------------- FINAL CHART --------------------
st.subheader("📈 Historical Prices + 5-Day Forecast")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df.index[-100:], df["Close"].values[-100:], label="Actual Price (Last 100 Days)", color="blue")
ax2.plot(future_df["Date"], future_df["Predicted Close"], label="Predicted Future Price", color="orange", marker="o")
ax2.set_title(f"{ticker}: Next 5-Day Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Forecast complete!")

