"""Backtest entrypoint: load configs, run strategy per ticker/interval, plot results."""

from __future__ import annotations

import argparse
import datetime
import json
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from strategies.supply_and_demand import (
    SupplyDemandStrategy,
    detect_supply_demand_zones,
    zones_to_frame,
)
from utils.yfinance_client import YFinanceClient


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

MA_WINDOWS = (5, 10, 20, 30, 48, 60, 120, 240)


def estimate_warmup_bars(
    *,
    ma_windows: tuple[int, ...] = MA_WINDOWS,
    macd_slow: int = 26,
    atr_window: int = 14,
    multiplier: int = 3,
) -> int:
    return int(max(max(ma_windows), macd_slow, atr_window) * multiplier)


def _align_tz(ts, tz) -> pd.Timestamp:
    """Localize/convert a Timestamp to `tz` (no-op when tz is None)."""
    ts = pd.Timestamp(ts)
    if tz is None:
        return ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def _slice_view(data: pd.DataFrame, start, end) -> pd.DataFrame:
    tz = getattr(data.index, "tz", None)
    return data.loc[_align_tz(start, tz):_align_tz(end, tz)]


def _filter_trades_by_view(trades_df: pd.DataFrame, start, end) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    entry_times = pd.to_datetime(trades_df["entry_time"])
    tz = getattr(entry_times.dt, "tz", None)
    start_ts, end_ts = _align_tz(start, tz), _align_tz(end, tz)
    return trades_df[(entry_times >= start_ts) & (entry_times <= end_ts)].copy()


def _filter_zones_by_view(zones_frame: pd.DataFrame, start, end) -> pd.DataFrame:
    if zones_frame.empty:
        return zones_frame
    tz = zones_frame["start"].dt.tz
    start_ts, end_ts = _align_tz(start, tz), _align_tz(end, tz)
    return zones_frame[(zones_frame["start"] >= start_ts) & (zones_frame["end"] <= end_ts)]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_history(history, title: str = "Balance Over Time") -> None:
    timestamps, values = zip(*history)
    plt.figure(figsize=(12, 6))
    plt.step(timestamps, values, where="post")
    plt.title(title)
    plt.xlabel("Time")
    plt.grid()
    plt.show()


def plot_pnl_vs_confidence(
    trades_df: pd.DataFrame, title: str = "PnL vs Zone Strength Confidence"
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(trades_df["confidence"], trades_df["pnl"], alpha=0.7)
    plt.title(title)
    plt.xlabel("Zone Strength Confidence")
    plt.ylabel("PnL ($)")
    plt.grid()
    plt.show()


# ---------------------------------------------------------------------------
# Account simulation
# ---------------------------------------------------------------------------

def _trade_direction(trade: pd.Series) -> int:
    for col in ("direction", "side", "type"):
        if col in trade.index:
            return 1 if str(trade[col]).lower().startswith("b") else -1
    return 1


def _build_event_log(trades_df: pd.DataFrame) -> list[tuple[pd.Timestamp, str, Any]]:
    """Return chronologically-ordered (timestamp, action, trade_id) tuples.

    Exits are resolved before entries that share the same timestamp so a
    position closed and opened on the same bar frees its margin first.
    """
    events: list[tuple[pd.Timestamp, str, Any]] = []
    for idx, trade in trades_df.iterrows():
        for action in ("entry", "exit"):
            ts = trade.get(f"{action}_time")
            if pd.notna(ts):
                events.append((pd.Timestamp(ts), action, idx))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "exit" else 1))
    return events


def simulate_account(
    data: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    balance_unit: str = "USD",
    leverage: int = 100,
    initial_balance: float = 1000,
    risk_percentage: float = 0.03,
    title: str = "Over Time",
) -> None:
    if trades_df.empty:
        print("No trades to simulate.")
        return

    events = _build_event_log(trades_df)
    if not events:
        print("Trades do not contain entry/exit timestamps.")
        return

    balance = initial_balance
    open_positions: dict[Any, dict[str, float]] = {}
    history: list[dict[str, Any]] = []

    def snapshot(ts, note: str) -> None:
        used_margin = sum(p["margin"] for p in open_positions.values())
        total_risk = sum(p["risk"] for p in open_positions.values())
        equity = balance - total_risk  # ignore unrealized PnL for simplicity
        history.append({
            "timestamp": ts,
            "balance": balance,
            "equity": equity,
            "used_margin": used_margin,
            "free_margin": balance - used_margin,
            "margin_level": (equity / used_margin * 100) if used_margin > 0 else float("inf"),
            "open_trades": len(open_positions),
            "note": note,
        })

    snapshot(data.index[0], "start")

    for ts, action, idx in events:
        trade = trades_df.loc[idx]
        entry_price = float(trade["entry_price"])
        direction = _trade_direction(trade)

        if action == "entry":
            stop_loss = float(trade.get("stop_loss", entry_price))
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
        snapshot(data.index[-1], "mark open positions")

    history_df = pd.DataFrame(history).sort_values("timestamp")
    print(f"Total Profit: {(balance - initial_balance):.2f} {balance_unit}\n")

    equity_curve = [(row.timestamp, row.equity) for row in history_df.itertuples()]
    plot_history(equity_curve, title="Equity " + title)


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------

def _load_intervals(test_infos: dict) -> tuple[list[str], bool]:
    cfg = test_infos["interval"]
    intervals = cfg if isinstance(cfg, list) else [cfg]
    use_multi = len(intervals) == 4
    return (intervals if use_multi else [intervals[0]]), use_multi


def _view_start_for(end_date, base_duration: pd.Timedelta, idx: int, use_multi: bool):
    """Each subsequent interval doubles the displayed history window."""
    duration = base_duration * (2 ** idx) if use_multi else base_duration
    start_ts = pd.Timestamp(end_date) - duration
    return start_ts.date() if hasattr(start_ts, "date") else start_ts


def _run_interval(
    *,
    client: YFinanceClient,
    target: str,
    ticker: str,
    spread: float,
    interval: str,
    start_date,
    end_date,
    view_start_date,
    config_path: str,
) -> dict[str, Any]:
    """Fetch data, detect zones, run strategy and return assembled artifacts."""
    data_ = client.get_between(
        ticker, view_start_date, end_date,
        interval=interval, warmup_bars=estimate_warmup_bars(),
    )
    data_view = _slice_view(data_, start_date, end_date)

    zones_in_window = _filter_zones_by_view(
        zones_to_frame(detect_supply_demand_zones(data_, config_path=config_path)),
        start_date, end_date,
    )
    print(f"Detected {len(zones_in_window)} zones for {ticker} {interval}:")
    print(zones_in_window)

    strategy = SupplyDemandStrategy(
        ticker=ticker, client=client,
        start=start_date, end=end_date,
        interval=interval, spread=spread,
        config_path=config_path,
    )
    strategy.run(visualize=False)

    trades_view = _filter_trades_by_view(strategy.get_trades_df(), start_date, end_date)
    if trades_view.empty:
        print(f"No trades for {target} {interval}.")
    else:
        print(f"Trades for {target} {interval}:")
        print(trades_view)

    return {
        "data": data_,
        "data_view": data_view,
        "zones": strategy.get_zones(),
        "trades_view": trades_view,
        "view_start": view_start_date,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest your strategy.")
    parser.add_argument("targets", type=str, nargs="+", default=["usdchf"],
                        help="Ticker symbol(s) to backtest on")
    parser.add_argument("--plot", action="store_true", help="Whether to plot candles")
    parser.add_argument("--config_path", type=str, default="configs/sad_params.json",
                        help="Path to strategy config file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open("configs/tickers.json") as fh:
        ticker_names = json.load(fh)
    with open("configs/test_infos.json") as fh:
        test_infos = json.load(fh)

    start_date = datetime.date.fromisoformat(test_infos["start_date"])
    end_date = datetime.date.fromisoformat(test_infos["end_date"])
    intervals_to_run, use_multi_interval_plot = _load_intervals(test_infos)

    trades_by_interval: dict[str, pd.DataFrame] = {i: pd.DataFrame() for i in intervals_to_run}
    tickers_with_trades: dict[str, set[str]] = {i: set() for i in intervals_to_run}
    data_view_by_interval: dict[str, pd.DataFrame] = {}

    base_duration = pd.Timestamp(end_date) - pd.Timestamp(start_date)
    client = YFinanceClient()

    for target in args.targets:
        ticker_info = ticker_names.get(target.lower())
        if ticker_info is None:
            print(f"Ticker {target} not found in config, skipping.")
            continue
        ticker, spread = ticker_info["ticker"], ticker_info["spread"]

        interval_views: list[dict[str, Any]] = []
        for idx, interval in enumerate(intervals_to_run):
            view_start_date = _view_start_for(
                end_date, base_duration, idx, use_multi_interval_plot,
            )
            result = _run_interval(
                client=client, target=target, ticker=ticker, spread=spread,
                interval=interval, start_date=start_date, end_date=end_date,
                view_start_date=view_start_date, config_path=args.config_path,
            )
            data_view_by_interval[interval] = result["data_view"]
            trades_view = result["trades_view"]

            if not use_multi_interval_plot and args.plot:
                client.plot_candlestick(
                    result["data"], zones=result["zones"], trades_df=trades_view,
                    display_start=start_date, display_end=end_date,
                )

            if not trades_view.empty:
                simulate_account(
                    result["data_view"], trades_view,
                    title=f"Over Time for {target} ({interval})",
                )
                trades_by_interval[interval] = pd.concat(
                    [trades_by_interval[interval], trades_view]
                )
                tickers_with_trades[interval].add(target)
            else:
                print("No trades executed, skipping balance simulation.")

            interval_views.append({
                "interval": interval,
                "title": f"{ticker} {interval}",
                "data": result["data"],
                "zones": result["zones"],
                "trades_df": trades_view,
                "display_start": view_start_date,
                "display_end": end_date,
            })

        if use_multi_interval_plot and args.plot:
            client.plot_candlestick_grid(
                interval_views, display_start=start_date, display_end=end_date,
            )

    # Aggregated equity curve: only meaningful when 2+ tickers actually traded
    # on the same interval (otherwise it duplicates the per-ticker plot above).
    if len(args.targets) > 1:
        for interval, trades_all in trades_by_interval.items():
            if trades_all.empty or len(tickers_with_trades[interval]) < 2:
                continue
            sorted_trades = trades_all.sort_values(by="exit_time").reset_index(drop=True)
            simulate_account(
                data_view_by_interval[interval], sorted_trades,
                title=f"Over Time for {', '.join(args.targets)} ({interval})",
            )


if __name__ == "__main__":
    main()
