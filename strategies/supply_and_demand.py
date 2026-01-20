import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

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
    base_lengths: Sequence[int] = (2, 3, 4, 5, 6),
    base_atr_multiplier: float = 2.5,
    move_atr_multiplier: float = 2.0,
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

            breakout_bar_threshold = atr_mean * move_atr_multiplier
            prior_move = _has_prior_move(close, base_start, trend_lookback, breakout_bar_threshold)
            post_move = _has_post_move(close, base_end, confirm_lookahead, breakout_bar_threshold)

            if prior_move is None or post_move is None:
                continue
            breakout_bar = df.iloc[base_end]
            breakout_range = float(breakout_bar[close_col]) - float(breakout_bar[open_col])
            breakout_close_move = float(close.iloc[base_end] - close.iloc[base_end - 1])
            if abs(breakout_close_move) < breakout_bar_threshold and breakout_range < breakout_bar_threshold:
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

            # Avoid duplicate
            if any(
                (z.zone_type == zone_type)
                and abs(midpoint - z.midpoint) <= (z_strength_atr * min_separation_atr)
                for z, z_strength_atr in ((z, df.loc[z.start:z.end, "atr"].mean()) for z in zones)
            ):
                continue

            strength = abs(post_move) / breakout_bar_threshold
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

if __name__ == "__main__":
    ticker = "CL=F"
    client = YFinanceClient()
    startdate = datetime.date(2025, 11, 11)
    enddate = datetime.date(2025, 11, 13)

    data_ = client.get_between(
        ticker,
        startdate,
        enddate,
        interval="1h",
    )

    back_testing_data = client.get_between(
        ticker,
        datetime.date(2025, 11, 14),
        datetime.date(2026, 1, 10),
        interval="1h",
    )
    zones = detect_supply_demand_zones(data_)
    zones_frame = zones_to_frame(zones)

    print(f"Detected {len(zones_frame)} zones for {ticker} (past three months, 1H):")
    print(zones_frame)

    client.plot_candlestick(data_, zones=zones)