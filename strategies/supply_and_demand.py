import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import pandas as pd
import pdb

from utils.yfinance_client import YFinanceClient

ZoneType = Literal["supply", "demand"]


@dataclass
class SupplyDemandZoneCandle:
    zone_type: ZoneType
    start: pd.Timestamp
    end: pd.Timestamp
    upper: float
    lower: float
    pattern: str
    strength: float
    is_broken: bool = False
    broken_time: Optional[pd.Timestamp] = None

    @property
    def midpoint(self) -> float:
        return (self.upper + self.lower) / 2


def _atr(
    df: pd.DataFrame,
    *,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 14,
) -> pd.Series:
    """Average True Range."""
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift()
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=1).mean()


def _has_prior_move(close: pd.Series, start: int, lookback: int, threshold: float) -> Optional[float]:
    """Return signed move into the base if it clears the threshold."""
    if start - lookback < 0:
        return None
    move = close.iloc[start - 1] - close.iloc[start - lookback]
    if abs(move) < threshold:
        return None
    return move


def _has_post_move(close: pd.Series, end: int, lookahead: int, threshold: float) -> Optional[float]:
    """Return signed move out of the base if it clears the threshold."""
    if end + lookahead >= len(close):
        return None
    move = close.iloc[end + lookahead - 1] - close.iloc[end - 1]
    if abs(move) < threshold:
        return None
    return move


def _zones_too_close(
    zones: Sequence[SupplyDemandZoneCandle],
    zone_type: ZoneType,
    midpoint: float,
    df: pd.DataFrame,
    min_separation_atr: float,
) -> bool:
    for zone in zones[-2:]:
        atr_slice = df.loc[zone.start:zone.end, "atr"]
        if atr_slice.empty:
            continue
        zone_atr = float(atr_slice.mean())
        if pd.isna(zone_atr):
            continue
        if zone.zone_type == zone_type and abs(midpoint - zone.midpoint) <= zone_atr * min_separation_atr:
            return True
    return False


def _duplicate_zone(
    zones: Sequence[SupplyDemandZoneCandle],
    zone_high: float,
    zone_low: float,
) -> bool:
    if not zones:
        return False
    last_zone = zones[-1]
    return zone_high == last_zone.upper and zone_low == last_zone.lower


def _consecutive_conflict(
    zones: Sequence[SupplyDemandZoneCandle],
    df: pd.DataFrame,
    base_start: int,
    base_end: int,
) -> bool:
    if len(zones) < 2:
        return False
    idx = df.index
    last_zone = zones[-1]
    penultimate = zones[-2]
    return (
        last_zone.start == idx[base_start]
        or penultimate.end == idx[base_end - 1]
        or penultimate.end == idx[base_start]
    )


def detect_supply_demand_zones(
    price_df: pd.DataFrame,
    *,
    base_lengths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    base_atr_multiplier: float = 5.0,
    move_atr_multiplier_prior: float = 0.5,
    move_atr_multiplier_post: float = 1.5,
    trend_lookback: int = 6,
    confirm_lookahead: int = 6,
    min_separation_atr: float = 0.25,
    high_col: str = "High",
    low_col: str = "Low",
    open_col: str = "Open",
    close_col: str = "Close",
    zone_buffer_percentage: float = 0.0,
) -> List[SupplyDemandZoneCandle]:
    """
    Identify supply/demand zones using a simplified rally/drop-base-rally/drop playbook.

    The logic mirrors the YouTube walkthrough: look for a tight base (small ATR range),
    ensure price moved into the base, and confirm a meaningful move away.
    """
    required_cols = {high_col, low_col, close_col}
    missing = required_cols - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    df = price_df.copy().sort_index()
    df["atr"] = _atr(df, high_col=high_col, low_col=low_col, close_col=close_col)
    close = df[close_col]
    zones: List[SupplyDemandZoneCandle] = []

    max_base = max(base_lengths)
    if len(df) < max_base + confirm_lookahead + trend_lookback:
        return zones

    for base_start in range(trend_lookback, len(df) - confirm_lookahead - min(base_lengths)):
        for base_len in base_lengths:
            base_end = base_start + base_len
            if base_end + confirm_lookahead > len(df):
                break

            window = df.iloc[base_start:base_end]
            atr_mean = window["atr"].mean()
            if atr_mean == 0:
                continue

            base_range = window[high_col].max() - window[low_col].min()
            if base_range > atr_mean * base_atr_multiplier:
                continue

            breakout_bar_threshold_prior = atr_mean * move_atr_multiplier_prior
            breakout_bar_threshold_post = atr_mean * move_atr_multiplier_post

            prior_move = _has_prior_move(close, base_start, trend_lookback, breakout_bar_threshold_prior)
            post_move = _has_post_move(close, base_end, confirm_lookahead, breakout_bar_threshold_post)

            if prior_move is None or post_move is None:
                continue

            breakout_bar = df.iloc[base_end]
            breakout_range = float(breakout_bar[close_col]) - float(breakout_bar[open_col])
            breakout_close_move = float(close.iloc[base_end] - close.iloc[base_end - 1])
            if abs(breakout_close_move) < breakout_bar_threshold_post and breakout_range < breakout_bar_threshold_post:
                continue

            pattern: Optional[str] = None
            zone_type: Optional[ZoneType] = None
            if prior_move > 0 and post_move < 0:
                pattern, zone_type = "rally-base-drop", "supply"
            elif prior_move < 0 and post_move > 0:
                pattern, zone_type = "drop-base-rally", "demand"
            elif prior_move < 0 and post_move < 0:
                pattern, zone_type = "drop-base-drop", "supply"
            elif prior_move > 0 and post_move > 0:
                pattern, zone_type = "rally-base-rally", "demand"

            if pattern is None or zone_type is None:
                continue

            zone_high = float(df.iloc[base_end - 1][high_col])
            zone_low = float(df.iloc[base_end - 1][low_col])
            if zone_type == "supply": zone_high = max(zone_high, float(df.iloc[base_end][high_col]))
            if zone_type == "demand":  zone_low = min(zone_low, float(df.iloc[base_end][low_col]))
            midpoint = (zone_high + zone_low) / 2

            if _zones_too_close(zones, zone_type, midpoint, df, min_separation_atr):
                continue
            if _duplicate_zone(zones, zone_high, zone_low):
                continue
            if _consecutive_conflict(zones, df, base_start, base_end):
                continue
            original_range = zone_high - zone_low
            strength = abs(breakout_range) / abs(original_range)
            if strength <= 0.6: continue
            zones.append(
                SupplyDemandZoneCandle(
                    zone_type=zone_type,
                    start=df.index[base_start],
                    end=df.index[base_end - 1],
                    upper=zone_high + original_range * zone_buffer_percentage,
                    lower=zone_low - original_range * zone_buffer_percentage,
                    pattern=pattern,
                    strength=float(strength),
                )
            )

    return zones


def zones_to_frame(zones: List[SupplyDemandZoneCandle]) -> pd.DataFrame:
    """Convert detected zones into a friendly DataFrame."""
    return pd.DataFrame(
        [
            {
                "type": z.zone_type,
                "start": z.start,
                "end": z.end,
                "upper": z.upper,
                "lower": z.lower,
                "pattern": z.pattern,
                "strength": z.strength,
            }
            for z in zones
        ]
    )


class SupplyDemandStrategy():
    def __init__(
        self,
        *,
        ticker: str,
        client: YFinanceClient,
        start: datetime.date,
        end: datetime.date,
        interval: str = "4h",
        spread: float = 0.001,
        time_buffer: int = 3,
        risk_reward_ratio: float = 2.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ticker = ticker
        self.client = client
        self.start = start
        self.end = end
        self.interval = interval
        self._data: Optional[pd.DataFrame] = None
        self._zones: List[SupplyDemandZoneCandle] = []
        self.time_buffer = time_buffer
        self.spread = spread
        self.risk_reward_ratio = risk_reward_ratio
        self.trades_df: pd.DataFrame = pd.DataFrame()
        self.trade_count: int = 0
        self.total_profit: float = 0.0

    def load_data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self.client.get_between(
                self.ticker,
                self.start,
                self.end,
                interval=self.interval,
            )
        return self._data

    def get_trades_df(self) -> pd.DataFrame:
        return self.trades_df.copy()

    def analyze(self) -> pd.DataFrame:
        data = self.load_data()
        self._zones = detect_supply_demand_zones(data)
        return zones_to_frame(self._zones)

    def run(self, visualize: bool = False) -> pd.DataFrame:
        self.analyze()
        trades_df = self.place_trades(self._zones)
        self.trades_df = trades_df
        if visualize:
            self.client.plot_candlestick(self.load_data(), zones=self._zones, trades_df=trades_df)
        return trades_df

    def place_trades(self, zones: Optional[Sequence[SupplyDemandZoneCandle]] = None) -> pd.DataFrame:
        data = self.load_data().sort_index().copy()
        if zones is None:
            if not self._zones:
                self._zones = detect_supply_demand_zones(data)
            zones_list: Sequence[SupplyDemandZoneCandle] = self._zones
        else:
            zones_list = list(zones)

        trades: List[dict] = []
        for zone in zones_list:
            trade = self._enter_trade_from_zone(data, zone)
            if trade:
                zone.is_broken = True
                zone.broken_time = trade["exit_time"]
                trades.append(trade)

        trades_df = pd.DataFrame(trades)
        self.trade_count = len(trades_df)
        self.total_profit = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        self._log_trade_summary(trades_df)
        return trades_df

    def _enter_trade_from_zone(
        self,
        data: pd.DataFrame,
        zone: SupplyDemandZoneCandle,
    ) -> Optional[dict]:
        if zone.end not in data.index:
            return None
        start_idx = data.index.get_loc(zone.end) + self.time_buffer
        if start_idx >= len(data):
            return None
        for idx in range(start_idx, len(data)):
            row = data.iloc[idx]
            if not self._zone_touched(row, zone):
                continue
            return self._build_trade_record(data, idx, zone)
        return None

    @staticmethod
    def _zone_touched(row: pd.Series, zone: SupplyDemandZoneCandle) -> bool:
        high_price = float(row["High"])
        low_price = float(row["Low"])
        return low_price <= zone.upper and high_price >= zone.lower

    def _build_trade_record(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        zone: SupplyDemandZoneCandle,
    ) -> Optional[dict]:
        direction = "short" if zone.zone_type == "supply" else "long"
        entry_price = self._entry_price(zone, direction)
        stop_loss, take_profit = self._risk_targets(zone, direction, entry_price)
        if stop_loss is None or take_profit is None:
            return None

        exit_price, exit_time, outcome = self._walk_forward(
            data,
            entry_idx,
            direction,
            entry_price,
            stop_loss,
            take_profit,
        )
        pnl = entry_price - exit_price if direction == "short" else exit_price - entry_price

        return {
            "zone_type": zone.zone_type,
            "entry_time": data.index[entry_idx],
            "exit_time": exit_time,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_price": exit_price,
            "pnl": pnl,
            "confidence": zone.strength,
            "outcome": outcome,
        }

    def _entry_price(self, zone: SupplyDemandZoneCandle, direction: str) -> float:
        price = zone.upper if direction == "long" else zone.lower
        return price + self.spread if direction == "long" else price - self.spread

    def _risk_targets(
        self,
        zone: SupplyDemandZoneCandle,
        direction: str,
        entry_price: float,
    ) -> tuple[Optional[float], Optional[float]]:
        if direction == "short":
            stop_loss = float(zone.upper)
            risk = stop_loss - entry_price
            if risk <= 0:
                return None, None
            take_profit = entry_price - self.risk_reward_ratio * risk
        else:
            stop_loss = float(zone.lower)
            risk = entry_price - stop_loss
            if risk <= 0:
                return None, None
            take_profit = entry_price + self.risk_reward_ratio * risk
        return stop_loss, take_profit

    def _walk_forward(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[float, pd.Timestamp, str]:
        default_exit_price = float(data.iloc[-1]["Close"])
        default_exit_time = data.index[-1]
        default_outcome = (
            "win" if (direction == "short" and default_exit_price <= entry_price)
            or (direction == "long" and default_exit_price >= entry_price)
            else "loss"
        )

        for future_idx in range(entry_idx + 1, len(data)):
            future = data.iloc[future_idx]
            future_high = float(future["High"])
            future_low = float(future["Low"])
            timestamp = data.index[future_idx]

            if direction == "short":
                if future_high >= stop_loss:
                    return stop_loss, timestamp, "loss"
                if future_low <= take_profit:
                    return take_profit, timestamp, "win"
            else:
                if future_low <= stop_loss:
                    return stop_loss, timestamp, "loss"
                if future_high >= take_profit:
                    return take_profit, timestamp, "win"

        return default_exit_price, default_exit_time, default_outcome

    def _log_trade_summary(self, trades_df: pd.DataFrame) -> None:
        print(trades_df)
        print(f"Total profit: {self.total_profit:.2f}")
        if trades_df.empty:
            print("Win rate: N/A")
        else:
            win_rate = (trades_df["outcome"] == "win").mean()
            print(f"Win rate: {win_rate:.2%}")


if __name__ == "__main__":
    import json

    with open("configs/tickers.json") as fh:
        target_names = json.load(fh)
    with open("configs/test_infos.json") as fh:
        test_infos = json.load(fh)

    start_date = datetime.date.fromisoformat(test_infos["start_date"])
    end_date = datetime.date.fromisoformat(test_infos["end_date"])
    interval = test_infos["interval"]

    target = target_names["usdchf"]
    ticker = target["ticker"]
    client = YFinanceClient()

    data_ = client.get_between(ticker, start_date, end_date, interval=interval)
    zones = detect_supply_demand_zones(data_)
    zones_frame = zones_to_frame(zones)

    print(f"Detected {len(zones_frame)} zones for {ticker} (past three months, {interval}):")
    print(zones_frame)
    client.plot_candlestick(data_, zones=zones)

    SupplyDemandStrategy(
        ticker=ticker,
        client=client,
        start=start_date,
        end=end_date,
        interval=interval,
        spread=target["spread"],
    ).run()