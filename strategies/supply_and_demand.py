import datetime
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
import pdb
ZoneType = Literal["supply", "demand"]


@dataclass
class SupplyDemandZone:
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
    base_atr_multiplier: float = 1.0,
    move_atr_multiplier: float = 1.0,
    trend_lookback: int = 3,
    confirm_lookahead: int = 4,
    min_separation_atr: float = 0.25,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
) -> List[SupplyDemandZone]:
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
    zones: List[SupplyDemandZone] = []

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
            
            threshold = atr_mean * move_atr_multiplier
            prior_move = _has_prior_move(close, base_start, trend_lookback, threshold)
            post_move = _has_post_move(close, base_end, confirm_lookahead, threshold)

            if prior_move is None or post_move is None:
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
                for z, z_strength_atr in ((z, df.loc[z.start:z.end, "atr"].mean()) for z in zones)
            ):
                continue

            strength = abs(post_move) / threshold
            zones.append(
                SupplyDemandZone(
                    zone_type=zone_type,
                    start=df.index[base_start],
                    end=df.index[base_end - 1],
                    upper=upper,
                    lower=lower,
                    pattern=pattern,
                    strength=float(strength),
                )
            )

    return zones


def zones_to_frame(zones: List[SupplyDemandZone]) -> pd.DataFrame:
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


def find_zones_for_ticker(
    ticker: str,
    *,
    months: int = 3,
    interval: str = "4h",
    auto_adjust: bool = True,
    **zone_kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch recent data via yfinance and return detected supply/demand zones.

    Returns (price_df, zones_df).
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=months * 30)
    ticker_obj = yf.Ticker(ticker)
    price = ticker_obj.history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    zones = detect_supply_demand_zones(price, **zone_kwargs)
    return price, zones_to_frame(zones)


if __name__ == "__main__":
    ticker = "USDCHF=X"
    prices, zones = find_zones_for_ticker(ticker)
    print(f"Detected {len(zones)} zones for {ticker} (past three months, 4H):")
    print(zones.tail(20))
