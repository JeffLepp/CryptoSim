# ========== Install dependencies (only need to run once in terminal) ==========
# pip install yfinance numpy pandas scikit-learn tensorflow

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import matplotlib.pyplot as plt

if __name__ == "main":
    # ===================== Configuration ===================== #
    TICKER = "AAPL"          # Stock ticker (e.g. TSLA)
    START_DATE = "2015-01-01"
    END_DATE = "2025-11-01"

    WINDOW_SIZE = 60         # Number of past days to predict the next day
    TEST_RATIO = 0.2         # Last 20% as test set


    def create_lstm_positions_from_df(
        df,
        window_size=WINDOW_SIZE,
        test_ratio=TEST_RATIO,
        epochs=10,
        batch_size=32,
        max_pos=10,
        up_threshold=0.01,
        down_threshold=-0.01,
    ):
        """
        Train the existing LSTM model on df['Price'] (BTC data passed from main.py),
        then use next-step predictions to build a 'position' column.

        Returns a copy of df with a new 'position' column.
        """

        # ---- 1) Prepare price data ----
        df = df.copy()
        if "Price" not in df.columns:
            raise ValueError("DataFrame must contain a 'Price' column")

        prices = df["Price"].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(prices)

        # ---- 2) Create sequences and train the same kind of model as below ----
        X, y = create_sequences(scaled_close, window_size)  # reuses your existing function

        split_index = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
        )

        # ---- 3) Use predictions to build positions ----
        df["position"] = 0  # start flat

        for i in range(window_size, len(df)):
            prev_pos = df.at[i - 1, "position"]

            window = scaled_close[i - window_size : i]
            x_input = window.reshape((1, window_size, 1))

            pred_scaled = model.predict(x_input, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0, 0]

            current_price = df.at[i, "Price"]
            if current_price <= 0:
                df.at[i, "position"] = prev_pos
                continue

            predicted_return = (pred_price - current_price) / current_price

            if predicted_return > up_threshold:
                new_pos = min(prev_pos + 1, max_pos)
            elif predicted_return < down_threshold:
                new_pos = max(prev_pos - 1, 0)
            else:
                new_pos = prev_pos

            df.at[i, "position"] = new_pos

        return df


    # ===================== 1. Download Data ===================== #
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)

    # Use only Close price (can later add Volume, MA, RSI, etc.)
    df = df[["Close"]].dropna()

    print(df.head())
    print(df.tail())
    print("Total samples:", len(df))

    # ===================== 2. Normalization ===================== #
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(df[["Close"]].values)

    # ===================== 3. Create Sliding Window Sequences ===================== #
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])  # Past window_size days
            y.append(data[i, 0])                # Value on day i
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_close, WINDOW_SIZE)
    print("X shape (samples, timesteps):", X.shape)
    print("y shape (samples,):", y.shape)

    # Reshape for LSTM input: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # ===================== 4. Train/Test Split ===================== #
    split_index = int(len(X) * (1 - TEST_RATIO))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # ===================== 5. Build LSTM Model ===================== #
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))  # Predict next normalized closing price

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # ===================== 6. Train Model ===================== #
    history = model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ===================== 7. Predict & Inverse Transform ===================== #
    y_pred_scaled = model.predict(X_test)

    y_test_scaled_2d = y_test.reshape(-1, 1)
    y_pred_2d = y_pred_scaled.reshape(-1, 1)

    y_test_inv = scaler.inverse_transform(y_test_scaled_2d)
    y_pred_inv = scaler.inverse_transform(y_pred_2d)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    # ===================== 8. Visualization: True vs Prediction ===================== #
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="True Close")
    plt.plot(y_pred_inv, label="Predicted Close")
    plt.title(f"{TICKER} - LSTM Test Set Prediction Performance")
    plt.xlabel("Sample index (in chronological order)")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/test_set_prediction.png", dpi=300)
    plt.close()
    print("Saved: results/test_set_prediction.png")

    # ===================== 9. Next-Day Forecast ===================== #
    print("Predicting the next day's closing price...")

    # Get the last WINDOW_SIZE days
    last_window = scaled_close[-WINDOW_SIZE:]
    print("Last window shape before reshape:", last_window.shape)

    # LSTM expects shape (1, timesteps, features)
    last_window = last_window.reshape((1, WINDOW_SIZE, 1))
    print("Last window shape for prediction:", last_window.shape)

    next_day_scaled = model.predict(last_window)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0, 0]
    print(f"Predicted next-day closing price: {next_day_price:.2f}")

    last_real_price = float(df["Close"].iloc[-1])
    print(f"Last real closing price: {last_real_price:.2f}")
    print(f"Predicted closing price for next trading day: {next_day_price:.2f}")

    if next_day_price > last_real_price:
        print("→ Model suggests a short-term **increase**.")
    else:
        print("→ Model suggests a **decline or consolidation**.")

    # Plot recent trend + prediction point
    recent_prices = df["Close"].iloc[-WINDOW_SIZE:]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(recent_prices)), recent_prices, label="Recent True Close")
    plt.scatter(len(recent_prices), next_day_price, marker="o", label="Predicted Price")
    plt.title(f"{TICKER} - Next-Day Prediction")
    plt.xlabel("Days back")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/next_day_prediction.png", dpi=300)
    plt.close()
    print("Saved: results/next_day_prediction.png")

    # ===================== 10. Recursive Forecasting for N Future Days ===================== #
    N_FUTURE_DAYS = 7

    future_predictions = []
    current_window = scaled_close[-WINDOW_SIZE:].copy()

    for _ in range(N_FUTURE_DAYS):
        x_input = current_window.reshape((1, WINDOW_SIZE, 1))
        pred_scaled = model.predict(x_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        future_predictions.append(pred_price)

        current_window = np.vstack([current_window[1:], pred_scaled])  # shift window

    print(f"Recursive prediction for next {N_FUTURE_DAYS} days:")
    for i, p in enumerate(future_predictions, 1):
        print(f"  Day {i}: {p:.2f}")

    # Plot future forecast
    plt.figure(figsize=(8, 4))
    days_ahead = range(1, N_FUTURE_DAYS + 1)
    plt.plot(days_ahead, future_predictions, marker="o", label="Predicted Close")
    plt.title(f"{TICKER} - Next {N_FUTURE_DAYS} Days Forecast")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Closing Price")
    plt.xticks(days_ahead)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/recursive_next_week.png", dpi=300)
    plt.close()
    print("Saved: results/recursive_next_week.png")
