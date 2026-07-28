"""1-minute technical analysis + a Warrior-style momentum score.

Given a 1-minute OHLCV frame (as returned by
``PolygonClient.aggregates(..., timespan="minute")``) this module computes the
indicators a momentum day-trader watches intraday and rolls them into a single
0-100 ``momentum_score``. A higher score means "the move still has fuel".

Indicators
----------
* EMA 9 / 20 / 48 / 60 / 200  (fast-to-slow trend stack)
* MACD (12, 26, 9): diff, signal (dea), histogram
* VWAP  (session-anchored, reset each trading day)
* Volume + a rolling average for relative volume

Score components (configurable weights)
--------------------------------------
* ``rvol``        - relative volume vs. its recent average (the bigger the
                    better; saturates so a 50x spike doesn't dwarf everything).
* ``macd_above``  - MACD diff above zero (零轴之上) and histogram rising.
* ``green_peak``  - the highest-volume bar of the session is an up (green) bar.
* ``above_vwap``  - last price trades above VWAP.
* ``ema_stack``   - bullish EMA alignment (9 > 20 > 48 > 60 > 200).

The weights live in ``DEFAULT_WEIGHTS`` and can be overridden from config.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


EMA_WINDOWS = (9, 20, 48, 60, 200)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "rvol": 30.0,
    "macd_above": 20.0,
    "green_peak": 15.0,
    "above_vwap": 20.0,
    "ema_stack": 15.0,
}


@dataclass
class MomentumScore:
    """Result of scoring one symbol's 1-min frame."""

    symbol: Optional[str]
    score: float                       # 0-100
    components: Dict[str, float] = field(default_factory=dict)   # weighted contribution
    raw: Dict[str, float] = field(default_factory=dict)          # 0-1 sub-scores
    snapshot: Dict[str, float] = field(default_factory=dict)     # latest indicator values

    def as_dict(self) -> Dict[str, float]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 1),
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "raw": {k: round(v, 3) for k, v in self.raw.items()},
            "snapshot": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in self.snapshot.items()},
        }


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------
def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, re-anchored at the start of each day."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical * df["Volume"]
    if isinstance(df.index, pd.DatetimeIndex):
        # Anchor sessions to the US trading day even when the index is displayed
        # in another timezone (e.g. Europe/Zurich), so after-hours bars are not
        # split across two local calendar days.
        if df.index.tz is not None:
            day = df.index.tz_convert("America/New_York").normalize()
        else:
            day = df.index.normalize()
        cum_pv = pv.groupby(day).cumsum()
        cum_vol = df["Volume"].groupby(day).cumsum()
    else:  # pragma: no cover - non-datetime index fallback
        cum_pv = pv.cumsum()
        cum_vol = df["Volume"].cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_indicators(
    df: pd.DataFrame,
    *,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rvol_window: int = 20,
) -> pd.DataFrame:
    """Return a copy of ``df`` with EMA/MACD/VWAP/volume columns added."""
    out = df.copy()
    close = out["Close"]

    for window in EMA_WINDOWS:
        out[f"ema{window}"] = close.ewm(span=window, adjust=False, min_periods=1).mean()

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    out["macd_diff"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_diff"].ewm(span=macd_signal, adjust=False).mean()
    out["macd_hist"] = out["macd_diff"] - out["macd_signal"]

    out["vwap"] = _session_vwap(out)
    out["vol_avg"] = out["Volume"].rolling(rvol_window, min_periods=1).mean()
    out["rvol"] = out["Volume"] / out["vol_avg"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _clamp01(value: float) -> float:
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def momentum_score(
    df: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    rvol_saturation: float = 10.0,
    already_has_indicators: bool = False,
) -> MomentumScore:
    """Compute a 0-100 momentum score from a 1-min OHLCV frame.

    ``rvol_saturation`` is the relative-volume level that earns the full rvol
    sub-score (e.g. 10 => 10x average volume scores 1.0). ``weights`` overrides
    ``DEFAULT_WEIGHTS``; the total need not sum to 100, the score is normalized
    to the supplied weights so disabling a component re-weights the rest.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if df is None or df.empty:
        return MomentumScore(symbol=symbol, score=0.0)

    ind = df if already_has_indicators else compute_indicators(df)
    last = ind.iloc[-1]

    # --- sub-scores in [0, 1] ------------------------------------------------
    raw: Dict[str, float] = {}

    rvol = last.get("rvol", np.nan)
    raw["rvol"] = _clamp01(rvol / rvol_saturation) if rvol == rvol else 0.0

    # MACD above zero line, with a bonus weighting for a rising histogram.
    diff = last.get("macd_diff", 0.0)
    hist = last.get("macd_hist", 0.0)
    above = 1.0 if diff > 0 else 0.0
    rising = 1.0 if hist > 0 else 0.0
    raw["macd_above"] = 0.6 * above + 0.4 * rising

    # Is the highest-volume bar of the session a green (up-close) candle?
    peak_idx = ind["Volume"].idxmax()
    peak = ind.loc[peak_idx]
    raw["green_peak"] = 1.0 if float(peak["Close"]) >= float(peak["Open"]) else 0.0

    # Price above VWAP, scaled by how far above (caps at +2%).
    vwap = last.get("vwap", np.nan)
    if vwap == vwap and vwap > 0:
        edge = (float(last["Close"]) - float(vwap)) / float(vwap)
        raw["above_vwap"] = _clamp01(0.5 + edge / 0.02 * 0.5) if edge > -0.02 else 0.0
    else:
        raw["above_vwap"] = 0.0

    # Bullish EMA stack: count adjacent fast>slow relationships.
    emas = [last.get(f"ema{w}", np.nan) for w in EMA_WINDOWS]
    pairs = list(zip(emas[:-1], emas[1:]))
    valid = [(a, b) for a, b in pairs if a == a and b == b]
    if valid:
        ordered = sum(1 for a, b in valid if a >= b)
        raw["ema_stack"] = ordered / len(valid)
    else:
        raw["ema_stack"] = 0.0

    # --- weighted total ------------------------------------------------------
    total_weight = sum(weights.get(k, 0.0) for k in raw)
    components = {k: weights.get(k, 0.0) * raw[k] for k in raw}
    score = (sum(components.values()) / total_weight * 100.0) if total_weight else 0.0

    snapshot = {
        "price": float(last["Close"]),
        "vwap": float(vwap) if vwap == vwap else None,
        "rvol": float(rvol) if rvol == rvol else None,
        "macd_diff": float(diff),
        "macd_hist": float(hist),
        **{f"ema{w}": (float(last.get(f"ema{w}")) if last.get(f"ema{w}") == last.get(f"ema{w}") else None)
           for w in EMA_WINDOWS},
    }
    return MomentumScore(
        symbol=symbol, score=score, components=components, raw=raw, snapshot=snapshot,
    )
