import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import pandas as pd
from utils.yfinance_client import YFinanceClient
import pdb
import random

class DummyStrategy():
    def __init__(self, stop_loss: float, take_profit: float, data: Optional[pd.DataFrame] = None):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.data = data
    
    def run(self,num_trades: int = 100):
        print("Running dummy strategy... (this is just a placeholder for actual logic)")
        self.place_trade(num_trades=num_trades)

    def get_trades_df(self) -> pd.DataFrame:
        return self.trades_df
    
    def place_trade(self, num_trades: int) -> List[dict]:
        # This dummy strategy just randomly pick some start data and randomly simulates trades with fixed stop loss and take profit levels.
        trades = []
        
        for i in range(num_trades):  # Simulate a trade every 10 rows
            if i * 10 >= len(self.data):
                break
            # Randomly select an entry point
            # Ensure there's at least one data point left to check for exit
            if len(self.data) < 2:
                break
            
            entry_index = random.randint(0, len(self.data) - 2)
            
            entry_row = self.data.iloc[entry_index]
            entry_price = entry_row["Close"]
            timestamp = entry_row.name
            
            action = random.choice(["buy", "sell"])
            exit_price = None
            
            if action == "buy":
                stop_loss_price = entry_price * (1 - self.stop_loss)
                take_profit_price = entry_price * (1 + self.take_profit)
                
                # Look at subsequent data points to find an exit
                for j in range(entry_index + 1, len(self.data)):
                    future_row = self.data.iloc[j]
                    if future_row["Low"] <= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = future_row.name
                        break
                    if future_row["High"] >= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = future_row.name
                        break
                
                if exit_price is not None:
                    pnl = exit_price - entry_price
                
            else: # action == "sell"
                stop_loss_price = entry_price * (1 + self.stop_loss)
                take_profit_price = entry_price * (1 - self.take_profit)

                # Look at subsequent data points to find an exit
                for j in range(entry_index + 1, len(self.data)):
                    future_row = self.data.iloc[j]
                    if future_row["High"] >= stop_loss_price:
                        exit_price = stop_loss_price
                        exit_time = future_row.name
                        break
                    if future_row["Low"] <= take_profit_price:
                        exit_price = take_profit_price
                        exit_time = future_row.name
                        break
                
                if exit_price is not None:
                    pnl = entry_price - exit_price

            # If no exit condition was met by the end of the data, skip this trade
            if exit_price is None:
                continue
            
            trades.append({
                "timestamp": timestamp,
                "action": action,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "pnl": pnl,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "outcome": "win" if pnl > 0 else "loss",
            })
                
        trades_df = pd.DataFrame(trades)
        self.trade_count = int(len(trades_df))
        self.total_profit = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        self.trades_df = trades_df
        print(trades_df)
        print(f"Total profit: {self.total_profit:.2f}")
        return trades_df
    
if __name__ == "__main__":
    client = YFinanceClient()
    start_date = datetime.date(2025, 11, 10)
    end_date = datetime.date(2025, 11, 13)

    data_ = client.get_between(
        "AAPL",
        start_date,
        end_date,
        interval="1m",
    )

    strategy = DummyStrategy(stop_loss=0.02, take_profit=0.04)
    trades = strategy.place_trade(data_)
    print(trades)