import pandas as pd
import numpy as np
from pathlib import Path

from tax_class import TaxRule, Lot, get_tax_rate, match_lots, fifo_strategy, lifo_strategy, hifo_strategy, output_tax_report
from test import run_test

# Ensure you have loaded BTC data in a new folder named 'data' if you are running this for the first time.
# Due to it's size, it cannot be included in the repo directly, use the link below to find the dataset.
# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/versions/416?resource=download

FILE_NAME = "btcusd_1-min_data.csv" # 7291838 available rows, each representing a minute of BTC price data
SCALEDOWN_FACTOR = 30 # Large dataset, if set to 30 we will consider BTC price once every 30 mins

# Loading the BTC price data from CSV file
def load_price_data(file_name = FILE_NAME):

    # Search through folder 'data' and import dataframe
    file_path = 'data/' + file_name

    df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and i % SCALEDOWN_FACTOR != 0)

    # UNIX time is a little confusing, so we convert to datetime and sort by it
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
    df = df.sort_values("Timestamp").reset_index(drop=True)

    print("Loaded data with", len(df), "rows")
    return df


# Clean the data and add simple moving average indicators
def preprocess_data(df):
    df = df.copy()

    # Handle missing values in price column
    price_col = "Close" 
    df["Price"] = df[price_col].replace(0, np.nan)           # replace 0 values with NaN
    df["Price"] = df["Price"].interpolate(method='linear') # finally fill the missing values

    # Add returns in log scale
    df["log_return"] = np.log(df["Price"]).diff()            # find returns (log makes compounding linear)

    # Add some indicators - SMA means simple moving average
    df["SMA_short"] = df["Price"].rolling(window=30, min_periods=30).mean()   # short-term simple moving average (30 mins)
    df["SMA_long"] = df["Price"].rolling(window=100, min_periods=100).mean()  # long-term simple moving average (100 mins)

    return df


# Next we need some strategy to make a historical portfolio, unless later we decide to pass into it a portfolio
def create_portfolio_signals(df, max_pos=10, buy_threshold = 0.05, sell_threshold = -0.05):

    df = df.copy()
    df["position"] = 0

    if "log_return" not in df.columns:
        raise ValueError("DataFrame must contain 'log_return' column")

    df["momentum"] = df["log_return"].rolling(window=30, min_periods=30).mean()

    for i in range(1, len(df)):

        # Get position and current indicators
        prev_pos = df.at[i-1, "position"]
        price = df.at[i, "Price"]
        sma_long = df.at[i, "SMA_long"]
        momentum = df.at[i, "momentum"]

        # Hold if momentum/SMAlong isnt ready
        if pd.isna(sma_long) or pd.isna(momentum):
            df.at[i, "position"] = prev_pos
            continue

        # Find relative Deviance
        relative_dev = (price - sma_long) / sma_long

        # Act based upon deviance/momentum and thresholds
        if relative_dev > buy_threshold and momentum > 0:
            new_pos = min(prev_pos + 1, max_pos)
        elif relative_dev < sell_threshold and momentum < 0:
            new_pos = max(prev_pos - 1, 0)
        else:
            new_pos = prev_pos

        df.at[i, "position"] = new_pos

    return df

# Create buy/sell transactions from portfolio signals
def create_transactions_from_signals(df, trade_size = .1, fee_rate = .001):
    df = df.copy()
    df["pos_change"] = df["position"].diff().fillna(0).fillna(0)

    records = []
    it = 1

    for idx, change in df["pos_change"].items():
        if change == 0:
            continue

        # Reading portfolio signals
        if change > 0:
            trade_type = "BUY"
        elif change < 0:
            trade_type = "SELL"
        else:
            continue

        trade_amount = abs(change) * trade_size
        price = df.at[idx, "Price"]
        time = df.at[idx, "Timestamp"]
        fee = trade_amount * price * fee_rate

        # Records for tax calculation
        record = {
            "trade_id": it,
            "ticker": "BTCUSD",
            "timestamp": time,
            "trade_type": trade_type,
            "trade_amount": trade_amount,
            "price": price,
            "fee": fee
        }
        records.append(record)
        it += 1

    trades_df = pd.DataFrame(records)

    trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
    return trades_df

def main():

    df = load_price_data(FILE_NAME)
    df = preprocess_data(df)

    trade_size = 0.1
    fee_rate = .001

    buy_threshold = 0.05
    sell_threshold = -0.05
    max_pos = 10

    df = create_portfolio_signals(df, max_pos, buy_threshold, sell_threshold)
    trades_df = create_transactions_from_signals(df, trade_size, fee_rate)

    print("number of trades:", len(trades_df))

    taxRules = TaxRule(short_term_rate=0.3, long_term_rate=0.15, threshold_days=365)

    fifo_tax = fifo_strategy(trades_df, taxRules)
    fifo_summary = output_tax_report(fifo_tax)

    lifo_tax = lifo_strategy(trades_df, taxRules)
    lifo_summary = output_tax_report(lifo_tax)

    hifo_tax = hifo_strategy(trades_df, taxRules)
    hifo_summary = output_tax_report(hifo_tax)

    # Details tax reports if we find a clever way to demonstrate them visually
    fifo_tax.to_csv(f"output/fifo_tax_thr{buy_threshold}_pos{max_pos}_scale{SCALEDOWN_FACTOR}.csv", index=False)
    lifo_tax.to_csv(f"output/lifo_tax_thr{buy_threshold}_pos{max_pos}_scale{SCALEDOWN_FACTOR}.csv", index=False)
    hifo_tax.to_csv(f"output/hifo_tax_thr{buy_threshold}_pos{max_pos}_scale{SCALEDOWN_FACTOR}.csv", index=False)

    summary_df = pd.DataFrame([
        {"strategy": "FIFO", **fifo_summary},
        {"strategy": "LIFO", **lifo_summary},
        {"strategy": "HIFO", **hifo_summary},
    ])

    # Main thing to look at result wise when plotting
    summary_df.to_csv("output/summary.csv", index=False)

main()

# Expected output for   buy/sell thresholds at 5%, 
#                       max_position 10, 
#                       trade size 0.1 BTC, 
#                       fee rate 0.1%, 
#                       with scaledown_factor 30:

# Loaded data with 243061 rows
# number of trades: 3218
# === TAX REPORT ===
# Total Realized Gain: 95414.53
# Total Taxes:        58294.15
# Total After-Tax Gain: 37120.38
# === TAX REPORT ===
# Total Realized Gain: 95414.53
# Total Taxes:        58205.24
# Total After-Tax Gain: 37209.29
# === TAX REPORT ===
# Total Realized Gain: 95414.53
# Total Taxes:        58207.16
# Total After-Tax Gain: 37207.37