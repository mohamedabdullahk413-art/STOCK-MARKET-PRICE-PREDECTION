# apple_stock_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# Streamlit page setup
st.set_page_config(page_title="Apple Stock Prediction", layout="wide")
st.title(" Apple (AAPL) Stock Price Prediction")

# -------------------- LOAD DATA --------------------
st.sidebar.header("Data Range")

start_date = "2024-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

st.sidebar.write(f"**Start Date:** {start_date}")
st.sidebar.write(f"**End Date:** {end_date}")

st.write("### Loading Apple Stock Data...")
df = yf.download("AAPL", start=start_date, end=end_date)[["Open", "Close"]]
st.write(f"Data from **{start_date}** to **{end_date}**:")
st.dataframe(df.tail())

# -------------------- PREPROCESSING --------------------
data = df[["Close"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

# Create time sequences
def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i - seq_length:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# -------------------- MODEL --------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
early_stop = EarlyStopping(monitor="loss", patience=5)

with st.spinner("Training model... Please wait ⏳"):
    model.fit(X_train, y_train, batch_size=32, epochs=20, callbacks=[early_stop], verbose=0)

# -------------------- PREDICTION --------------------
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------- VISUALIZATION --------------------
st.subheader("📊 Actual vs Predicted Closing Prices")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, color="blue", label="Actual Price")
ax.plot(predicted_prices, color="red", label="Predicted Price")
ax.set_title("AAPL Stock Price Prediction (Test Data)")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# -------------------- FUTURE PREDICTION --------------------
st.subheader("Predicting the Next 5 Business Days")

last_60_days = scaled_data[-60:]
future_predictions = []

# Generate predictions for next 5 business days
for _ in range(5):
    X_future = np.reshape(last_60_days, (1, 60, 1))
    pred_price = model.predict(X_future)
    future_predictions.append(pred_price[0, 0])
    last_60_days = np.append(last_60_days[1:], pred_price)
    last_60_days = np.reshape(last_60_days, (60, 1))

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate next 5 business days (skip weekends)
future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=5)

# Create DataFrame for display
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close": future_predictions.flatten()
})

st.write("### 📅 5-Business-Day Price Forecast")
st.dataframe(future_df)

# -------------------- FINAL CHART --------------------
st.subheader("📈 Historical + Future Predictions")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df.index[-100:], df["Close"].values[-100:], label="Actual Price (Last 100 Days)", color="blue")
ax2.plot(future_df["Date"], future_df["Predicted Close"], label="Predicted Future Price", color="orange", marker="o")
ax2.set_title("AAPL: Actual vs Next 5-Business-Day Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Forecast complete!")

# ----
