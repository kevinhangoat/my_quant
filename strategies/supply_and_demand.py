import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple
from strategies.strategy import BaseStrategy

import pandas as pd
from utils.yfinance_client import YFinanceClient
import pdb
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


def detect_supply_demand_zones(
    price_df: pd.DataFrame,
    *,
    base_lengths: Sequence[int] = (1, 2, 3, 4, 5, 6),
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

            upper = float(window[high_col].max())
            lower = float(window[low_col].min())
            midpoint = (upper + lower) / 2

            
            if any(
                (z.zone_type == zone_type)
                and abs(midpoint - z.midpoint) <= (z_strength_atr * min_separation_atr)
                for z, z_strength_atr in ((z, df.loc[z.start:z.end, "atr"].mean()) for z in zones[-2:])
            ):
                continue
            # Avoid duplicate
            if len(zones) >= 1 and float(df.iloc[base_end - 1][high_col]) == zones[-1].upper and float(df.iloc[base_end - 1][low_col]) == zones[-1].lower:
                continue
            strength = abs(post_move) / breakout_bar_threshold_post
            zones.append(
                SupplyDemandZoneCandle(
                    zone_type=zone_type,
                    start=df.index[base_start],
                    end=df.index[base_end - 1],
                    upper=float(df.iloc[base_end - 1][high_col]),
                    lower=float(df.iloc[base_end - 1][low_col]),
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
        time_buffer: int = 5,
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
        return self.trades_df if hasattr(self, "trades_df") else pd.DataFrame()

    def analyze(self) -> pd.DataFrame:
        data = self.load_data()
        self._zones = detect_supply_demand_zones(data)
        return zones_to_frame(self._zones)

    def run(self, visualize: bool = False) -> pd.DataFrame:
        zones_frame = self.analyze()
        trades_df = self.place_trades(self._zones)
        self.trades_df = trades_df
        if visualize:
            self.client.plot_candlestick(self.load_data(), zones=self._zones, trades_df=trades_df)

    def place_trades(self, zones: Optional[Sequence[SupplyDemandZoneCandle]] = None) -> pd.DataFrame:
        data = self.load_data().sort_index().copy()
        if zones is None:
            if not self._zones:
                self._zones = detect_supply_demand_zones(data)
            zones_list: List[SupplyDemandZoneCandle] = list(self._zones)
        else:
            zones_list = list(zones)

        trades: List[dict] = []
        for zone in zones_list:
            if zone.end not in data.index:
                continue
            start_idx = data.index.get_loc(zone.end) + self.time_buffer
            for idx in range(start_idx, len(data)):
                row = data.iloc[idx]
                close_price = float(row["Close"])
                high_price = float(row["High"])
                low_price = float(row["Low"])

                broken = (
                    zone.zone_type == "supply"
                    and close_price > zone.upper
                    or zone.zone_type == "demand"
                    and close_price < zone.lower
                )
                # if broken:
                #     break

                touched = low_price <= zone.upper and high_price >= zone.lower
                if not touched:
                    continue

                direction = "short" if zone.zone_type == "supply" else "long"
                # entry_price = float(min(max(close_price, zone.lower), zone.upper))
                entry_price = zone.upper if direction == "long" else zone.lower
                entry_price = entry_price + self.spread if direction == "long" else entry_price - self.spread
                if direction == "short":
                    stop_loss = float(zone.upper)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        break
                    take_profit = entry_price - 2 * risk
                else:
                    stop_loss = float(zone.lower)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        break
                    take_profit = entry_price + 2 * risk

                exit_price: float
                exit_time = data.index[idx]
                outcome = "open"
                for future_idx in range(idx + 1, len(data)):
                    future = data.iloc[future_idx]
                    future_high = float(future["High"])
                    future_low = float(future["Low"])
                    timestamp = data.index[future_idx]

                    if direction == "short":
                        if future_high >= stop_loss:
                            exit_price = stop_loss
                            exit_time = timestamp
                            outcome = "loss"
                            break
                        if future_low <= take_profit:
                            exit_price = take_profit
                            exit_time = timestamp
                            outcome = "win"
                            break
                    else:
                        if future_low <= stop_loss:
                            exit_price = stop_loss
                            exit_time = timestamp
                            outcome = "loss"
                            break
                        if future_high >= take_profit:
                            exit_price = take_profit
                            exit_time = timestamp
                            outcome = "win"
                            break
                    exit_price = float(data.iloc[-1]["Close"])
                    exit_time = data.index[-1]
                    if direction == "short":
                        outcome = "win" if exit_price <= entry_price else "loss"
                    else:
                        outcome = "win" if exit_price >= entry_price else "loss"

                pnl = entry_price - exit_price if direction == "short" else exit_price - entry_price
                trades.append(
                    {
                        "zone_type": zone.zone_type,
                        "entry_time": data.index[idx],
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
                )
                break
        trades_df = pd.DataFrame(trades)
        self.trade_count = int(len(trades_df))
        self.total_profit = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        print(trades_df)
        print(f"Total profit: {self.total_profit:.2f}")
        print(f"Win rate: {(trades_df['outcome'] == 'win').mean():.2%}" if not trades_df.empty else "Win rate: N/A")
        return trades_df


if __name__ == "__main__":
    import json
    target_names = json.load(open(f"configs/tickers.json"))
    test_infos = json.load(open(f"configs/test_infos.json"))
    start_date = datetime.date.fromisoformat(test_infos["start_date"])
    end_date = datetime.date.fromisoformat(test_infos["end_date"])
    interval = test_infos["interval"]

    target = target_names["usdchf"] 
    ticker = target["ticker"]
    client = YFinanceClient()
    startdate = datetime.date.fromisoformat(test_infos["start_date"])
    enddate = datetime.date.fromisoformat(test_infos["end_date"])

    data_ = client.get_between(
        ticker,
        startdate,
        enddate,
        interval=interval,
    )
    zones = detect_supply_demand_zones(data_)
    zones_frame = zones_to_frame(zones)

    print(f"Detected {len(zones_frame)} zones for {ticker} (past three months, {interval}):")
    print(zones_frame)
    client.plot_candlestick(data_, zones=zones)


    SupplyDemandStrategy(
        ticker=ticker,
        client=client,
        start=startdate,
        end=enddate,
        interval=interval,
        spread=target["spread"],
    ).run()