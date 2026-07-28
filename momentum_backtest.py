"""Market-wide momentum backtest over Polygon history.

This mirrors the **live screener**: for each historical trading day it scans the
*entire* US market, finds the symbols that would have triggered the screener
that day, then simulates the momentum trade on each.

Efficiency
----------
The whole-market scan costs **one** API call per day: Polygon's grouped-daily
endpoint (:meth:`PolygonClient.grouped_daily`) returns OHLCV for every ticker at
once - the historical analogue of the live full-market snapshot. The cheap
criteria (price range, day change, relative volume) filter ~8,000 tickers down
to a handful, and only those candidates pay for a 1-minute aggregate fetch.

Relative volume uses a rolling average daily volume built from a short warmup of
grouped-daily calls before the start date, then updated as each day is processed
(so no per-symbol history requests are needed).

Flow per day::

    grouped_daily(day)              # 1 call -> every ticker's OHLCV
      -> build StockSnapshot rows   # price, day change vs prev close, rvol
      -> cheap screener criteria    # SAME functions the live screener uses
      -> candidates (a handful)
          -> aggregates(minute)     # 1 call each, only for candidates
          -> detect_trigger intraday
          -> MomentumStrategy       # pullback entry, scaled exits, stop
      -> update rolling vol + prev close

Usage::

    export POLYGON_API_KEY=...
    python momentum_backtest.py --start 2026-06-01 --end 2026-06-27
    python momentum_backtest.py --start 2026-06-26 --end 2026-06-27 --plot
    # restrict the universe (optional) for a faster/cheaper run:
    python momentum_backtest.py --start 2026-06-01 --end 2026-06-27 --symbols AAPL TSLA
"""

import argparse
import datetime
import json
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from utils.polygon_client import PolygonClient, PolygonError
from utils.momentum import compute_indicators, momentum_score
from strategies.moemteum import MomentumStrategy, detect_trigger
from screener import StockSnapshot, select_criteria


DEFAULT_PARAMS_PATH = os.path.join("configs", "momentum_params.json")


# ---------------------------------------------------------------------------
# Universe scan (one grouped-daily call per day = whole market)
# ---------------------------------------------------------------------------
class VolumeHistory:
    """Rolling average daily volume per ticker (for relative volume)."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._vols: Dict[str, Deque[float]] = {}

    def average(self, ticker: str) -> Optional[float]:
        dq = self._vols.get(ticker)
        if not dq:
            return None
        return sum(dq) / len(dq)

    def update(self, ticker: str, volume: Optional[float]) -> None:
        if volume is None:
            return
        dq = self._vols.setdefault(ticker, deque(maxlen=self.window))
        dq.append(float(volume))


def _warmup_volume_history(
    polygon: PolygonClient, start: datetime.date, params: Dict[str, Any],
    *, include_otc: bool,
) -> "tuple[VolumeHistory, Dict[str, float]]":
    """Seed rolling volumes + previous closes from days before ``start``.

    Walks calendar days backward from the day before ``start`` until it has
    collected ``rvol_avg_window`` trading days (skipping weekends/holidays that
    return no grouped data). Returns the populated history and the most recent
    close per ticker (the "previous close" for the first backtested day).
    """
    window = int(params.get("rvol_avg_window", 20))
    history = VolumeHistory(window)
    prev_close: Dict[str, float] = {}

    collected: List[Dict[str, Dict[str, Any]]] = []
    day = start - datetime.timedelta(days=1)
    guard = 0
    while len(collected) < window and guard < window * 3 + 10:
        guard += 1
        try:
            grouped = polygon.grouped_daily(day, include_otc=include_otc)
        except PolygonError as exc:
            print(f"[warmup] {day} failed: {exc}")
            grouped = {}
        if grouped:
            collected.append(grouped)
        day -= datetime.timedelta(days=1)

    # Apply oldest-to-newest so the rolling window and prev_close are correct.
    for grouped in reversed(collected):
        for ticker, row in grouped.items():
            history.update(ticker, row.get("volume"))
            if row.get("close") is not None:
                prev_close[ticker] = float(row["close"])
    return history, prev_close


def scan_day(
    grouped: Dict[str, Dict[str, Any]],
    prev_close: Dict[str, float],
    history: VolumeHistory,
    params: Dict[str, Any],
    cheap_criteria,
) -> List[StockSnapshot]:
    """Apply the live screener's cheap criteria to a whole-market day."""
    survivors: List[StockSnapshot] = []
    for ticker, row in grouped.items():
        # Use the day's HIGH (not close): intraday the screener fires the moment
        # the price spikes to its peak, even if the stock fades back by close.
        high = row.get("high")
        pc = prev_close.get(ticker)
        if high is None or pc in (None, 0):
            continue
        snap = StockSnapshot(
            symbol=ticker,
            price=high,
            change_pct=(high - pc) / pc * 100.0,
            day_volume=row.get("volume"),
            avg_volume=history.average(ticker),
            prev_close=pc,
        )
        if all(fn(snap, params).passed for fn in cheap_criteria):
            survivors.append(snap)
    return survivors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_trades(symbol: str, day: datetime.date, trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        return
    show = trades_df.copy()
    for col in ("entry_price", "exit_price", "stop_loss", "take_profit", "pnl"):
        if col in show:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
    if "pnl_pct" in show:
        show["pnl_pct"] = (pd.to_numeric(show["pnl_pct"], errors="coerce") * 100).round(2)
    cols = [c for c in ("entry_time", "entry_price", "exit_time", "exit_price",
                        "pnl_pct", "num_legs") if c in show]
    print(show[cols].to_string(index=False))
    print("\n")


def _aggregate(all_trades: pd.DataFrame) -> None:
    if all_trades.empty:
        print("\nNo trades across the whole backtest.")
        return
    pnl_pct = pd.to_numeric(all_trades["pnl_pct"], errors="coerce").dropna()
    if pnl_pct.empty:
        print("\nNo completed trades to aggregate.")
        return
    wins = (pnl_pct > 0).sum()
    print("\n=== Aggregate ===")
    print(f"Trades        : {len(pnl_pct)}")
    print(f"Symbols       : {all_trades['symbol'].nunique()}")
    print(f"Win rate      : {wins / len(pnl_pct) * 100:.1f}%")
    print(f"Avg return    : {pnl_pct.mean() * 100:.2f}% per trade")
    print(f"Total return  : {pnl_pct.sum() * 100:.2f}% (1 unit risked each)")
    print(f"Best / worst  : {pnl_pct.max() * 100:.2f}% / {pnl_pct.min() * 100:.2f}%")


def _save_results(all_trades: pd.DataFrame, start: datetime.date,
                  end: datetime.date, *, out_dir: str = "back_testing") -> None:
    """Export the backtest trades to ``back_testing/<start>_<end>.csv``."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{start.isoformat()}_{end.isoformat()}.csv")
    all_trades.to_csv(path, index=False)
    print(f"\nSaved {len(all_trades)} trade row(s) to {path}")


def _plot_day(symbol: str, day: datetime.date, day_bars: pd.DataFrame,
              trades_df: pd.DataFrame) -> None:
    """Plot one session's 1-min candles + EMA stack + MACD + trade markers."""
    try:
        from utils.yfinance_client import YFinanceClient
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        print(f"[plot] unavailable: {exc}")
        return
    YFinanceClient().plot_candlestick(
        day_bars,
        trades_df=trades_df if not trades_df.empty else None,
        ma_type="ema",
        ma_windows=(9, 20, 48, 60, 200),
        show_macd=True,
        title=f"{symbol} {day} (1-min momentum)",
    )


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------
def run_backtest(
    polygon: PolygonClient,
    start: datetime.date,
    end: datetime.date,
    params: Dict[str, Any],
    *,
    symbols: Optional[List[str]] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Scan the market each day in [start, end] and simulate triggered trades."""
    include_otc = bool(params.get("include_otc", False))
    extended = params.get("extended_hours", True)
    max_candidates = int(params.get("max_candidates_per_day", 50))
    universe = {s.upper() for s in symbols} if symbols else None
    cheap_criteria = select_criteria(params, "cheap")

    history, prev_close = _warmup_volume_history(
        polygon, start, params, include_otc=include_otc,
    )

    all_rows: List[Dict[str, Any]] = []
    day = start
    while day <= end:
        try:
            grouped = polygon.grouped_daily(day, include_otc=include_otc)
        except PolygonError as exc:
            print(f"[scan] {day} failed: {exc}")
            grouped = {}
        if not grouped:
            day += datetime.timedelta(days=1)
            continue
        if universe is not None:
            grouped = {t: r for t, r in grouped.items() if t in universe}

        candidates = scan_day(grouped, prev_close, history, params, cheap_criteria)
        # Trade the strongest movers first; cap minute-bar fetches per day.
        candidates.sort(key=lambda s: (s.change_pct or 0.0), reverse=True)
        if max_candidates > 0:
            candidates = candidates[:max_candidates]

        if candidates:
            syms = ", ".join(s.symbol for s in candidates[:10])
            more = "" if len(candidates) <= 10 else f" (+{len(candidates) - 10} more)"
            print(f"\n[{day}] {len(candidates)} screener hit(s): {syms}{more}")

        for snap in candidates:
            rows = _simulate_symbol_day(
                polygon, snap, day, params, extended=extended, plot=plot,
            )
            all_rows.extend(rows)

        # Roll volumes + previous closes forward with today's data.
        for ticker, row in grouped.items():
            history.update(ticker, row.get("volume"))
            if row.get("close") is not None:
                prev_close[ticker] = float(row["close"])
        day += datetime.timedelta(days=1)

    return pd.DataFrame(all_rows)


def _simulate_symbol_day(
    polygon: PolygonClient,
    snap: StockSnapshot,
    day: datetime.date,
    params: Dict[str, Any],
    *,
    extended: bool,
    plot: bool,
) -> List[Dict[str, Any]]:
    """Fetch one candidate's 1-min bars, confirm the intraday trigger, simulate."""
    try:
        day_bars = polygon.aggregates(
            snap.symbol, day, day, multiplier=1, timespan="minute",
            include_extended_hours=extended,
        )
    except PolygonError as exc:
        print(f"  [minute] {snap.symbol} {day} failed: {exc}")
        return []
    if day_bars.empty:
        return []

    ind = compute_indicators(day_bars)
    trigger = detect_trigger(
        ind, prev_close=snap.prev_close, avg_volume=snap.avg_volume, params=params,
    )
    if trigger is None:
        return []

    score = momentum_score(ind, symbol=snap.symbol, already_has_indicators=True)
    print(f"  {snap.symbol:<6} trigger @ {trigger.time()} "
          f"| day chg {snap.change_pct:.0f}% | momentum {score.score:.0f}")

    strategy = MomentumStrategy(params)
    strategy.run(
        day_bars, symbol=snap.symbol, trigger_time=trigger,
        prev_close=snap.prev_close, avg_volume=snap.avg_volume,
    )
    trades_df = strategy.trades_df()
    _print_trades(snap.symbol, day, trades_df)
    if plot:
        _plot_day(snap.symbol, day, day_bars, trades_df)
    if trades_df.empty:
        return []

    rows = trades_df.to_dict("records")
    for row in rows:
        row["date"] = day.isoformat()
        row["triggered_at"] = trigger.isoformat()
        row["momentum_score"] = score.score
    return rows


def _load_params(path: Optional[str]) -> Dict[str, Any]:
    path = path or DEFAULT_PARAMS_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Market-wide momentum backtest (scans every ticker per day)",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Optional: restrict the scanned universe to these tickers")
    parser.add_argument("--params", default=None, help="Path to momentum params JSON")
    parser.add_argument("--plot", action="store_true", help="Plot each triggered session")
    parser.add_argument("--polygon-key", default=None, help="Override POLYGON_API_KEY")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        polygon = PolygonClient(api_key=args.polygon_key)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    params = _load_params(args.params)
    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)

    all_trades = run_backtest(
        polygon, start, end, params, symbols=args.symbols, plot=args.plot,
    )
    _aggregate(all_trades)
    _save_results(all_trades, start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
