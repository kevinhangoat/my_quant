import numpy as np
import pandas as pd
from typing import Iterable, Tuple


def vegas_tunnel(
    price_df: pd.DataFrame,
    *,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    ema_fast: int = 144,
    ema_slow: int = 169,
    atr_window: int = 14,
    channel_multipliers: Iterable[float] = (1.618, 2.618, 4.236),
) -> pd.DataFrame:
    """
    Compute Vegas Channel/Tunnel levels and a simple pullback signal.

    - Tunnel = EMA(144) and EMA(169) on the close.
    - Channels = EMA(169) +/- ATR * multipliers (default 1.618/2.618/4.236).
    - Signal logic (trend-following pullback):
        * Long when fast EMA > slow EMA and price pierces the slow EMA intrabar
          but closes back above the fast EMA (treat as a bounce off the tunnel).
        * Short when fast EMA < slow EMA and price pierces the slow EMA intrabar
          but closes back below the fast EMA.
    Returns a copy of the input DataFrame with indicator columns:
    ema_fast, ema_slow, atr, upper_<mult>, lower_<mult>, signal, signal_change.
    """
    required_cols = {close_col, high_col, low_col}
    missing = required_cols - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    df = price_df.copy().sort_index()
    close = df[close_col]

    df["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    high = df[high_col]
    low = df[low_col]
    prev_close = close.shift()
    true_range = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    df["atr"] = true_range.rolling(atr_window, min_periods=1).mean()

    def _label(mult: float) -> str:
        return str(mult).replace(".", "p")

    for mult in channel_multipliers:
        label = _label(mult)
        df[f"upper_{label}"] = df["ema_slow"] + df["atr"] * mult
        df[f"lower_{label}"] = df["ema_slow"] - df["atr"] * mult

    bias_long = df["ema_fast"] > df["ema_slow"]
    bias_short = df["ema_fast"] < df["ema_slow"]

    long_bounce = bias_long & (low <= df["ema_slow"]) & (close > df["ema_fast"])
    short_bounce = bias_short & (high >= df["ema_slow"]) & (close < df["ema_fast"])
    df["signal"] = np.select([long_bounce, short_bounce], [1, -1], default=0).astype(int)
    df["signal_change"] = df["signal"].diff().fillna(0).astype(int)

    return df
