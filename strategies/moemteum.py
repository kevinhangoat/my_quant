"""Momentum day-trading backtest strategy (Warrior-Trading style).

The plan being tested:

1. **Screener trigger** - intraday a symbol becomes a candidate once it is up
   big on the day (``min_change_pct``) on heavy relative volume
   (``min_relative_volume``). :func:`detect_trigger` finds the first 1-min bar
   where that happens, mirroring the live screener's cheap criteria.
2. **Entry** - after the trigger, buy when price *pulls back* to the EMA48 or
   EMA60 (the classic "buy the dip on the moving average" entry) while the
   trend is still up.
3. **Scaled exits** - sell a slice of the position at +10%, +30% and +60%.
4. **Stop** - if price drops ``stop_loss_pct`` (default 10%) below the entry,
   dump the remaining shares.

The strategy consumes a 1-minute OHLCV frame (from
``PolygonClient.aggregates``) and produces a ``trades_df`` whose columns line up
with ``back_testing.simulate_account`` and ``YFinanceClient.plot_candlestick``
(``entry_time``, ``entry_price``, ``exit_time``, ``exit_price``, ``pnl``,
``stop_loss``, ``take_profit``, ``side``).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.momentum import compute_indicators


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "min_change_pct": 30.0,         # trigger: up >= 30% on the day
    "min_relative_volume": 5.0,     # trigger: rvol >= 5x
    "pullback_emas": [48, 60],      # entry on a touch of either EMA
    "pullback_tolerance": 0.003,    # within 0.3% of the EMA counts as a touch
    "require_above_slow_ema": True, # only enter while close >= slowest pullback EMA
    "profit_targets": [             # (gain_fraction, position_fraction_to_sell)
        [0.10, 0.33],
        [0.30, 0.33],
        [0.60, 0.34],
    ],
    "stop_loss_pct": 0.10,          # exit remainder on a 10% drop from entry
    "trailing_stop": False,         # if True, stop trails the high-water mark
    "max_trades_per_session": 1,    # re-entries allowed after a full exit
}


@dataclass
class ExitLeg:
    time: pd.Timestamp
    price: float
    fraction: float       # fraction of the original position closed by this leg
    reason: str           # "target", "stop", "eod"


@dataclass
class MomentumTrade:
    symbol: Optional[str]
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    side: str = "long"
    legs: List[ExitLeg] = field(default_factory=list)

    @property
    def exit_time(self) -> Optional[pd.Timestamp]:
        return self.legs[-1].time if self.legs else None

    @property
    def exit_price(self) -> Optional[float]:
        """Position-fraction-weighted average exit price."""
        if not self.legs:
            return None
        total = sum(leg.fraction for leg in self.legs)
        if total <= 0:
            return None
        return sum(leg.price * leg.fraction for leg in self.legs) / total

    @property
    def take_profit(self) -> Optional[float]:
        targets = [leg.price for leg in self.legs if leg.reason == "target"]
        return max(targets) if targets else None

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0 or not self.legs:
            return 0.0
        return sum(
            (leg.price - self.entry_price) / self.entry_price * leg.fraction
            for leg in self.legs
        )

    @property
    def pnl(self) -> float:
        """Per-share PnL scaled by the fraction of the position each leg closed."""
        return sum((leg.price - self.entry_price) * leg.fraction for leg in self.legs)

    def to_row(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "num_legs": len(self.legs),
        }


# ---------------------------------------------------------------------------
# Trigger detection (historical screener)
# ---------------------------------------------------------------------------
def detect_trigger(
    minute_df: pd.DataFrame,
    *,
    prev_close: float,
    avg_volume: Optional[float],
    params: Dict[str, Any],
) -> Optional[pd.Timestamp]:
    """Return the first timestamp where the screener criteria are met intraday.

    Day change is measured from ``prev_close``; relative volume from the day's
    accumulated volume vs. ``avg_volume`` (skipped when avg_volume is missing).
    """
    if minute_df.empty or not prev_close:
        return None
    min_change = params.get("min_change_pct", 30.0)
    min_rvol = params.get("min_relative_volume", 5.0)

    day_change = (minute_df["Close"] - prev_close) / prev_close * 100.0
    # Group cumulative volume by the US trading day even when the index is shown
    # in another timezone, so after-hours bars stay within one session.
    if minute_df.index.tz is not None:
        session = minute_df.index.tz_convert("America/New_York").normalize()
    else:
        session = minute_df.index.normalize()
    cum_volume = minute_df["Volume"].groupby(session).cumsum()

    change_ok = day_change >= min_change
    if avg_volume and avg_volume > 0:
        rvol_ok = (cum_volume / avg_volume) >= min_rvol
        qualifying = minute_df.index[change_ok & rvol_ok]
    else:
        qualifying = minute_df.index[change_ok]
    return qualifying[0] if len(qualifying) else None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class MomentumStrategy:
    """Simulate the pullback-entry / scaled-exit plan on a 1-min frame."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.trades: List[MomentumTrade] = []

    # -- entry helpers ----------------------------------------------------
    def _is_pullback(self, bar: pd.Series) -> bool:
        tol = self.params["pullback_tolerance"]
        emas = self.params["pullback_emas"]
        low, close = float(bar["Low"]), float(bar["Close"])

        # Only trade while the trend stack is intact: the faster pullback EMA
        # (EMA48) must be above the slower one (EMA60). No long entries otherwise.
        if len(emas) >= 2:
            fast_w, slow_w = min(emas), max(emas)
            fast_val = bar.get(f"ema{fast_w}")
            slow_val = bar.get(f"ema{slow_w}")
            if fast_val != fast_val or slow_val != slow_val:  # NaN guard
                return False
            if float(fast_val) <= float(slow_val):
                return False

        touched = any(
            (f"ema{w}" in bar) and (bar[f"ema{w}"] == bar[f"ema{w}"]) and
            low <= float(bar[f"ema{w}"]) * (1 + tol)
            for w in emas
        )
        if not touched:
            return False
        if self.params.get("require_above_slow_ema", True):
            slow = max(emas)
            slow_val = bar.get(f"ema{slow}")
            if slow_val == slow_val and close < float(slow_val) * (1 - tol):
                return False
        return True

    # -- core simulation --------------------------------------------------
    def run(
        self,
        minute_df: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
        trigger_time: Optional[pd.Timestamp] = None,
        prev_close: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> List[MomentumTrade]:
        """Find trigger (if not given), simulate entries/exits, return trades."""
        if minute_df.empty:
            self.trades = []
            return self.trades
        ind = compute_indicators(minute_df)

        if trigger_time is None and prev_close is not None:
            trigger_time = detect_trigger(
                ind, prev_close=prev_close, avg_volume=avg_volume, params=self.params,
            )
        if trigger_time is None:
            self.trades = []
            return self.trades

        after = ind.loc[ind.index >= trigger_time]
        trades: List[MomentumTrade] = []
        max_trades = self.params.get("max_trades_per_session", 1)
        i = 0
        bars = list(after.itertuples(index=True))
        n = len(bars)
        while i < n and len(trades) < max_trades:
            # --- look for a pullback entry ---
            entry_idx = None
            while i < n:
                bar = after.loc[bars[i][0]]
                if self._is_pullback(bar):
                    entry_idx = i
                    break
                i += 1
            if entry_idx is None:
                break

            entry_time = bars[entry_idx][0]
            entry_price = float(after.loc[entry_time, "Close"])
            trade = self._simulate_trade(after, entry_idx, bars, entry_time, entry_price, symbol)
            trades.append(trade)
            # resume scanning after this trade's last leg
            if trade.legs:
                last_time = trade.legs[-1].time
                while i < n and bars[i][0] <= last_time:
                    i += 1
            else:
                i = entry_idx + 1

        self.trades = trades
        return trades

    def _simulate_trade(self, after, entry_idx, bars, entry_time, entry_price, symbol):
        stop_pct = self.params["stop_loss_pct"]
        trailing = self.params.get("trailing_stop", False)
        targets = [list(t) for t in self.params["profit_targets"]]
        remaining = 1.0
        high_water = entry_price
        stop_price = entry_price * (1 - stop_pct)
        trade = MomentumTrade(
            symbol=symbol, entry_time=entry_time, entry_price=entry_price,
            stop_loss=stop_price,
        )

        for j in range(entry_idx + 1, len(bars)):
            ts = bars[j][0]
            bar = after.loc[ts]
            high, low = float(bar["High"]), float(bar["Low"])

            if trailing:
                high_water = max(high_water, high)
                stop_price = max(stop_price, high_water * (1 - stop_pct))
                trade.stop_loss = stop_price

            # Stop check first (conservative: assume the worst within the bar).
            if low <= stop_price and remaining > 0:
                trade.legs.append(ExitLeg(ts, stop_price, remaining, "stop"))
                remaining = 0.0
                break

            # Profit targets (may fill several within one bar).
            for target in targets:
                gain, frac = target
                if frac <= 0:
                    continue
                tp_price = entry_price * (1 + gain)
                if high >= tp_price and remaining > 0:
                    fill = min(frac, remaining)
                    trade.legs.append(ExitLeg(ts, tp_price, fill, "target"))
                    remaining -= fill
                    target[1] = 0.0  # consumed
            if remaining <= 1e-9:
                break

        if remaining > 1e-9:
            last_ts = bars[-1][0]
            last_close = float(after.loc[last_ts, "Close"])
            trade.legs.append(ExitLeg(last_ts, last_close, remaining, "eod"))
        return trade

    # -- output -----------------------------------------------------------
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "symbol", "side", "entry_time", "entry_price", "exit_time",
                "exit_price", "stop_loss", "take_profit", "pnl", "pnl_pct", "num_legs",
            ])
        return pd.DataFrame([t.to_row() for t in self.trades])
