import datetime
from strategies.supply_and_demand import detect_supply_demand_zones, zones_to_frame, SupplyDemandStrategy
from strategies.dummy import DummyStrategy
from utils.yfinance_client import YFinanceClient
import matplotlib.pyplot as plt
import argparse
import pdb

import json
import pandas as pd

def plot_balance_history(balance_history, title="Simulated Balance Over Time"):
    timestamps, balances = zip(*balance_history)
    plt.figure(figsize=(12, 6))
    plt.step(timestamps, balances, where='post')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Balance ($)")
    plt.grid()
    plt.show()

def plot_pnl_vs_confidence(trades_df, title="PnL vs Zone Strength Confidence"):
    plt.figure(figsize=(10, 6))
    plt.scatter(trades_df["confidence"], trades_df["pnl"], alpha=0.7)
    plt.title(title)
    plt.xlabel("Zone Strength Confidence")
    plt.ylabel("PnL ($)")
    plt.grid()
    plt.show()
    
def simulate_balance(data, trades_df, contract_size = 100000, leverage = 100, initial_balance=1000, free_margin=8, title="Simulated Balance Over Time"):
    current_balance = initial_balance
    balance_history = [(data.index[0], current_balance)]
    sorted_trades = trades_df.sort_values(by="exit_time")
    for _, trade in sorted_trades.iterrows():
        pnl = trade["pnl"]
        entry_price = trade["entry_price"]
        margin = current_balance / (1+free_margin)
        real_pnl = pnl / entry_price * margin * leverage
        current_balance += real_pnl
        balance_history.append((trade["exit_time"], current_balance))
    plot_balance_history(balance_history, title=title)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Backtest your strategy.")
    argparser.add_argument("targets", type=str, nargs='+', default=["usdchf"], help="Ticker symbol(s) to backtest on")
    argparser.add_argument("--plot", action="store_true", help="Whether to plot candles")
    args = argparser.parse_args()

    
    ticker_names = json.load(open(f"configs/tickers.json"))
    test_infos = json.load(open(f"configs/test_infos.json"))
    start_date = datetime.date.fromisoformat(test_infos["start_date"])
    end_date = datetime.date.fromisoformat(test_infos["end_date"])
    interval = test_infos["interval"]
    trades_df_all = pd.DataFrame()
    
    for target in args.targets:
        ticker_info = ticker_names.get(target.lower())
        if ticker_info is None:
            print(f"Ticker {target} not found in config, skipping.")
            continue
        ticker = ticker_info["ticker"]
        spread = ticker_info["spread"]
        client = YFinanceClient()

        data_ = client.get_between(
            ticker,
            start_date,
            end_date,
            interval=interval,
        )
        
        zones = detect_supply_demand_zones(data_)
        zones_frame = zones_to_frame(zones)



        print(f"Detected {len(zones_frame)} zones for {ticker} (past three months, {interval}):")
        print(zones_frame)

        cur_strategy = SupplyDemandStrategy(
            ticker=ticker,
            client=client,
            start=start_date,
            end=end_date,
            interval=interval,
            spread=spread,
        )
        cur_strategy.run(visualize=args.plot)

        # Dummy strategy backtesting
        # cur_strategy = DummyStrategy(
        #     stop_loss = 0.01,
        #     take_profit = 0.02,
        #     data = data_
        # )
        # cur_strategy.run()

        trades_df = cur_strategy.get_trades_df()
        if not trades_df.empty:
            simulate_balance(data_, trades_df, title=f"Simulated Balance Over Time for {target}")
            trades_df_all = pd.concat([trades_df_all, trades_df])
            # plot_pnl_vs_confidence(trades_df)
        else:
            print("No trades executed, skipping balance simulation.")

if len(args.targets) > 1:
    sorted_trades_df_all = trades_df_all.sort_values(by="exit_time")
    simulate_balance(data_, sorted_trades_df_all, title=f"Simulated Balance Over Time for {', '.join(args.targets)}")