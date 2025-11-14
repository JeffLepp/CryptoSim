import pandas as pd
import numpy as np
from pathlib import Path

# Ensure you have loaded BTC data in a new folder named 'data' if you are running this for the first time.
# Due to it's size, it cannot be included in the repo directly, use the link below to find the dataset.
# https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/versions/416?resource=download

FILE_NAME = "btcusd_1-min_data.csv"
NUM_ROWS_TO_LOAD = 500000       # Limiting number of rows we import from file due to it's massive size


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
def create_portfolio(df):
    df = df.copy()
    df["position"] = 0   

    trending_up = (df["SMA_short"] > df["SMA_long"]) & (df["Price"] > 1.01 * df["SMA_long"])
    df["momentum"] = df["log_return"].rolling(window=30, min_periods=30).mean()

    df.loc[trending_up & (df["momentum"] > 0), "position"] = 1      # long position
    trades = df["position"].diff().abs()
    cost_per_trade = 0.001    # 0.1% trading cost per trade
    
    df["strategy_return"] = df["position"].shift(1) * df["log_return"]
    df["strategy_return"] -= trades * cost_per_trade

    return df


# Finally we compute performance of the strategy vs buy & hold
def compute_performance(df):
    df = df.copy()

    # Cumulative log returns of strategy
    df["cum_log_return_strat"] = df["strategy_return"].cumsum()
    df["equity_strat"] = np.exp(df["cum_log_return_strat"])   # starting at 1.0

    # For comparison: buy & hold (always position = 1)
    df["cum_log_return_buy_hold"] = df["log_return"].cumsum()
    df["equity_buy_hold"] = np.exp(df["cum_log_return_buy_hold"])

    final_equity_strat = df["equity_strat"].iloc[-1]
    final_equity_buy_hold = df["equity_buy_hold"].iloc[-1]

    print("=== PERFORMANCE over sample ===")
    print(f"Strategy total return:   {final_equity_strat - 1:.2%}")
    print(f"Buy & hold total return: {final_equity_buy_hold - 1:.2%}")

    return df


def main():

    df = load_price_data(FILE_NAME)
    df = preprocess_data(df)
    df = create_portfolio(df)
    df = compute_performance(df)

    # Here we have a synthetic example of any naive crypto strategy

    # We need now:
    
    # Estimate total capital gains

    # Simulate tax liability

    # Compare different trading rules

    # Evaluate whether your strategy is worth using at all

    # Feed into more advanced tax models (FIFO, LIFO, HIFO, etc.)

    # Build spreadsheet-ready tables

main()