from datetime import datetime, timedelta
import pandas as pd
from tax_class import TaxRule, fifo_strategy, lifo_strategy, hifo_strategy


data = [
    # Three buys at different prices
    {"trade_id": 1, "ticker": "BTCUSD",
        "timestamp": datetime(2020, 1, 1), "trade_type": "BUY",
        "trade_amount": 1.0, "price": 10_000, "fee": 0.0},
    {"trade_id": 2, "ticker": "BTCUSD",
        "timestamp": datetime(2020, 2, 1), "trade_type": "BUY",
        "trade_amount": 1.0, "price": 20_000, "fee": 0.0},
    {"trade_id": 3, "ticker": "BTCUSD",
        "timestamp": datetime(2020, 3, 1), "trade_type": "BUY",
        "trade_amount": 1.0, "price": 30_000, "fee": 0.0},

    # Sell part of the position later
    {"trade_id": 4, "ticker": "BTCUSD",
        "timestamp": datetime(2021, 3, 1), "trade_type": "SELL",
        "trade_amount": 2.0, "price": 40_000, "fee": 0.0},
]

def run_test(tax_rules):

    taxes = pd.DataFrame(data)

    fifo = fifo_strategy(taxes, tax_rules)
    lifo = lifo_strategy(taxes, tax_rules)
    hifo = hifo_strategy(taxes, tax_rules)

    print("FIFO:", fifo[["realized_gain", "taxes", "after_tax_gain"]].sum())
    print("LIFO:", lifo[["realized_gain", "taxes", "after_tax_gain"]].sum())
    print("HIFO:", hifo[["realized_gain", "taxes", "after_tax_gain"]].sum())