import datetime
from strategies.supply_and_demand import detect_supply_demand_zones, zones_to_frame, SupplyDemandStrategy
from strategies.dummy import DummyStrategy
from utils.yfinance_client import YFinanceClient
import matplotlib.pyplot as plt
import argparse
import pdb

import json
import pandas as pd

def plot_history(history, title="Balance Over Time"):
    timestamps, history_data = zip(*history)
    plt.figure(figsize=(12, 6))
    plt.step(timestamps, history_data, where='post')
    plt.title(title)
    plt.xlabel("Time")
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
    
def simulate_account(
    data,
    trades_df,
    balance_unit="USD",
    margin_level=6,
    contract_size=100000,
    leverage=100,
    initial_balance=1000,
    risk_percentage=0.06,
    title="Over Time",
):
    if trades_df.empty:
        print("No trades to simulate.")
        return

    events = []
    for idx, trade in trades_df.iterrows():
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        if pd.notna(entry_time):
            events.append((pd.Timestamp(entry_time), "entry", idx))
        if pd.notna(exit_time):
            events.append((pd.Timestamp(exit_time), "exit", idx))

    if not events:
        print("Trades do not contain entry/exit timestamps.")
        return

    events.sort(key=lambda e: (e[0], 0 if e[1] == "exit" else 1))

    balance = initial_balance
    open_positions = {}
    history = []

    def snapshot(ts, note):
        used_margin = sum(pos["margin"] for pos in open_positions.values())
        total_risk = sum(pos["risk"] for pos in open_positions.values())
        #Assuming there is no unrealized PnL for simplicity
        equity = balance - total_risk
        free_margin = balance - used_margin
        history.append(
            {
                "timestamp": ts,
                "balance": balance,
                "equity": equity,
                "used_margin": used_margin,
                "free_margin": free_margin,
                "margin_level": (equity / used_margin * 100) if used_margin > 0 else float('inf'),
                "open_trades": len(open_positions),
                "note": note,
            }
        )

    snapshot(data.index[0], "start")

    for ts, action, idx in events:
        trade = trades_df.loc[idx]
        entry_price = float(trade["entry_price"])
        direction_label = (
            trade["direction"]
            if "direction" in trade.index
            else trade["side"]
            if "side" in trade.index
            else trade["type"]
            if "type" in trade.index
            else "long"
        )
        direction = 1 if str(direction_label).lower().startswith("b") else -1

        if action == "entry":
            stop_loss = float(trade.get("stop_loss", entry_price))
            # margin = entry_price * balance * risk_percentage / (abs(entry_price - stop_loss) * leverage)
            margin = balance * risk_percentage
            notional = margin * leverage
            open_positions[idx] = {
                "entry_price": entry_price,
                "margin": margin,
                "notional": notional,
                "direction": direction,
                "risk": abs(entry_price - stop_loss) / entry_price * notional,
            }
            snapshot(ts, f"open #{idx}")
        else:
            position = open_positions.pop(idx, None)
            if position is None:
                continue
            pnl = float(trade.get("pnl", (float(trade["exit_price"]) - entry_price) * direction))
            realized = pnl / entry_price * position["notional"]
            balance += realized
            snapshot(ts, f"close #{idx} (PnL={realized:.2f})")

    if open_positions:
        final_ts = data.index[-1]
        snapshot(final_ts, "mark open positions")

    history_df = pd.DataFrame(history).sort_values("timestamp")
    print(history_df)

    equity_data = [(row.timestamp, row.equity) for row in history_df.itertuples()]
    plot_history(equity_data, title="Equity " + title)

    margin_level_data = [(row.timestamp, row.margin_level) for row in history_df.itertuples()]
    plot_history(margin_level_data, title="Margin Level " + title)

    margin_level_data = [(row.timestamp, row.open_trades) for row in history_df.itertuples()]
    plot_history(margin_level_data, title="Open Trades " + title)

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
            simulate_account(data_, trades_df, title=f"Over Time for {target}")
            trades_df_all = pd.concat([trades_df_all, trades_df])
            # plot_pnl_vs_confidence(trades_df)
        else:
            print("No trades executed, skipping balance simulation.")

    if len(args.targets) > 1:
        sorted_trades_df_all = trades_df_all.sort_values(by="exit_time").reset_index(drop=True)
        simulate_account(data_, sorted_trades_df_all, title=f"Over Time for {', '.join(args.targets)}")