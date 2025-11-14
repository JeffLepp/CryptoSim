from dataclasses import dataclass
from datetime import datetime

from copy import deepcopy
import pandas as pd

@dataclass

class TaxRule:
    short_term_rate: float
    long_term_rate: float
    threshold_days: int = 365

class Lot:
    buy_id: int
    coin: str
    buy_date: datetime
    buy_price: float
    amount: float
    remaining_amount: float

def get_tax_rate(holding_days: int, rule: TaxRule):
    if holding_days < rule.threshold_days:
        return rule.short_term_rate
    else:
        return rule.long_term_rate
    

def match_lots(transactions, tax_rule, lot_sort_key):
    tax = transactions.copy().reset_index(drop=True)
    open_lots = []
    tax_records = []

    for idx, row in tax.iterrows():
        trade_type = row
        date = row["timestamp"]
        coin = row["ticker"]
        price = row["price"]
        amount = row["trade_amount"]
        trade_id = row["trade_id"]

        if trade_type == "BUY":
            lot = Lot(
                buy_id=trade_id,
                coin=coin,
                buy_date=date,
                buy_price=price,
                amount=amount,
                remaining_amount=amount
            )
            open_lots.append(lot)

        elif trade_type == "SELL":
            qty_to_sell = amount
            proceeds = amount * price
            realized_gain = 0.0
            taxes = 0.0

            open_lots.sort(open_lots, key=lot_sort_key)

            for lot in open_lots:
                if qty_to_sell <= 0:
                    break
                if lot.remaining_amount <= 0:
                    continue

                used = min(lot.remaining_amount, qty_to_sell)
                cost_basis = used * lot.buy_price
                proceeds = used * price
                gain = proceeds - cost_basis

                holding_days = (date - lot.buy_date).days
                tax_rate = get_tax_rate(holding_days, tax_rule)

                tax = max(gain, 0,0) * tax_rate

                realized_gain += gain
                taxes += tax

                lot.remaining_amount -= used
                qty_to_sell -= used

            tax_records.append({
                "trade_id": trade_id,
                "ticker": coin,
                "timestamp": date,
                "quantity_sold": amount,
                "price": price,
                "proceeds": proceeds,
                "realized_gain": realized_gain,
                "taxes": taxes,
                "after_tax_gain": realized_gain - taxes
            })

    result = pd.Dataframe(tax_records).sort_values("timestamp").reset_index(drop=True)
    return result

# PUT STRATEGIES HERE
