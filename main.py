import pandas as pd
import numpy as np
from pathlib import Path

from tax_class import TaxRule, Lot, get_tax_rate, match_lots, fifo_strategy, lifo_strategy, hifo_strategy, output_tax_report
from test import run_test

# Ensure you have loaded BTC data in a new folder named 'data' if you are running this for the first time.
# Due to it's size, it cannot be included in the repo directly, use the link below to find the dataset.
# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/versions/416?resource=download

FILE_NAME = "btcusd_1-min_data.csv"
# 7291838 available rows, each representing a minute of BTC price data
NUM_ROWS_TO_LOAD = 80000       # Limiting number of rows we import from file due to it's massive size


# Loading the BTC price data from CSV file
def load_price_data(file_name = FILE_NAME):

    # Search through folder 'data' and import dataframe
    file_path = 'data/' + file_name

    df = pd.read_csv(file_path, nrows = NUM_ROWS_TO_LOAD)

    # UNIX time is a little confusing, so we convert to datetime and sort by it
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')
    df = df.sort_values("Timestamp").reset_index(drop=True)

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
def create_portfolio_signals(df, max_pos=3):
    df = df.copy()
    df["position"] = 0

    # Basic crossover signals
    diff = df["SMA_short"] - df["SMA_long"]
    prev_diff = diff.shift(1)

    # entry signal- short SMA crosses above long SMA
    buy_signal = (prev_diff <= 0) & (diff > 0)

    # exit signal- short SMA crosses below long SMA
    sell_signal = (prev_diff >= 0) & (diff < 0)
    df["momentum"] = df["log_return"].rolling(window=30, min_periods=30).mean()

    for i in range(1, len(df)):
        prev_pos = df.at[i-1, "position"]
        if buy_signal.iloc[i] and df["momentum"].iloc[i] > 0:
            df.at[i, "position"] = min(prev_pos + 1, max_pos)
        elif sell_signal.iloc[i]:
            df.at[i, "position"] = max(prev_pos - 1, 0)
        else:
            df.at[i, "position"] = prev_pos

    return df

def create_transactions_from_signals(df, trade_size = .1, fee_rate = .001):
    df = df.copy()
    df["pos_change"] = df["position"].diff().fillna(0).fillna(0)

    records = []
    it = 1

    for idx, change in df["pos_change"].items():
        if change == 0:
            continue

        # Portfolio buys at few strategic points
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
    df = create_portfolio_signals(df)

    trade_size = 0.1
    fee_rate = .001
    trades_df = create_transactions_from_signals(df, trade_size, fee_rate)

    print("number of trades:", len(trades_df))

    taxRules = TaxRule(short_term_rate=0.3, long_term_rate=0.15, threshold_days=365)

    # fifo_tax = fifo_strategy(trades_df, taxRules)
    # fifo_summary = output_tax_report(fifo_tax)

    # lifo_tax = lifo_strategy(trades_df, taxRules)
    # lifo_summary = output_tax_report(lifo_tax)

    # hifo_tax = hifo_strategy(trades_df, taxRules)
    # hifo_summary = output_tax_report(hifo_tax)

    run_test(taxRules)

main()