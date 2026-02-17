import datetime
from typing import Optional
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import pdb

class YFinanceClient:
    """Thin wrapper around yfinance for basic data retrieval."""

    def __init__(self, auto_adjust: bool = True, progress: bool = False) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    def get_history(
        self,
        ticker: str,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
        interval: str = "1d",
    ):
        """Fetch historical price data for a ticker."""
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
        )

    def get_1d(
        self,
        ticker: str,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
    ):
        """Return daily data for a ticker between start and end dates."""
        return self.get_history(ticker, start=start, end=end, interval="1d")

    def get_4h(
        self,
        ticker: str,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
    ):
        """Return 4-hour interval data for a ticker."""
        return self.get_history(ticker, start=start, end=end, interval="4h")

    def get_1h(
        self,
        ticker: str,
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
    ):
        """Return 1-hour interval data for a ticker."""
        return self.get_history(ticker, start=start, end=end, interval="1h")
        
    def get_past_three_years(self, ticker: str):
        """Convenience helper returning daily data for the last three years."""
        today = datetime.date.today()
        three_years_ago = today - datetime.timedelta(days=3 * 365)
        return self.get_history(ticker, start=three_years_ago, end=today, interval="1d")
    
    def get_past_three_months(self, ticker: str):
        """Convenience helper returning daily data for the last three months."""
        today = datetime.date.today()
        three_months_ago = today - datetime.timedelta(days=3 * 30)
        return self.get_4h(ticker, start=three_months_ago, end=today)

    def get_between(
        self,
        ticker: str,
        start: datetime.datetime,
        end: datetime.datetime,
        interval: str = "1m",
    ):
        """Return price data for a ticker between two datetimes."""
        if start >= end:
            raise ValueError("start must be earlier than end")
        return self.get_history(ticker, start=start, end=end, interval=interval)

    def plot_candlestick(
        self,
        data,
        zones = [],
        trades_df = None,
    ):
        """Plot a candlestick chart with optional supply/demand zones."""

        apds = []

        def _zones_to_addplots(zones, color):
            for zone in zones:
                start_time, broken_time, low, high, strength = zone.end, zone.broken_time, zone.lower, zone.upper, zone.strength
                index_tz = getattr(data.index, "tz", None)
                if index_tz is not None:
                    if start_time.tzinfo is None:
                        start_time = start_time.tz_localize(index_tz)
                    else:
                        start_time = start_time.tz_convert(index_tz)
                mask = data.index >= start_time
                if broken_time is not None:
                    mask &= data.index <= broken_time
                if not mask.any():
                    continue
                y_low = pd.Series(index=data.index, dtype="float64")
                y_high = y_low.copy()
                y_low.loc[mask] = low
                y_high.loc[mask] = high
                apds.append(
                    mpf.make_addplot(
                        y_high,
                        panel=0,
                        color=color,
                        alpha=0,
                        fill_between=dict(y1=y_high.values, y2=y_low.values, alpha=0.2, color=color),
                    )
                )
                if strength is not None and mask.any():
                    zone_indices = data.index[mask]
                    mid_idx = zone_indices[len(zone_indices) // 2]
                    strength_series = pd.Series(index=data.index, dtype="float64")
                    strength_series.loc[mid_idx] = low + (high - low) / 2
                    strength_label = f"{float(strength):.2f}" if strength is not None else ""
                    apds.append(
                        mpf.make_addplot(
                            strength_series,
                            panel=0,
                            type="scatter",
                            marker=f"${strength_label}$",
                            markersize=260,
                            color=color,
                        )
                    )
                apds.append(
                    mpf.make_addplot(
                    y_low,
                    panel=0,
                    color=color,
                    linestyle="--",
                    width=0.6,
                    )
                )
                apds.append(
                    mpf.make_addplot(
                    y_high,
                    panel=0,
                    color=color,
                    linestyle="--",
                    width=0.6,
                    )
                )
        if trades_df is not None and not trades_df.empty:
            for _, trade in trades_df.iterrows():
                entry_time = pd.Timestamp(trade["entry_time"])
                index_tz = getattr(data.index, "tz", None)
                if index_tz is not None:
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.tz_localize(index_tz)
                    else:
                        entry_time = entry_time.tz_convert(index_tz)

                nearest_idx = data.index.get_indexer([entry_time], method="nearest")
                if nearest_idx[0] == -1:
                    continue
                entry_idx = data.index[nearest_idx[0]]

                entry_price = trade["entry_price"]
                raw_side = str(trade.get("side", trade.get("direction", "buy"))).strip().lower()
                label = "BUY" if raw_side.startswith("l") else "SELL"
                marker = f"${label}$"
                color = "green" if label == "BUY" else "red"

                marker_series = pd.Series(index=data.index, dtype="float64")
                marker_series.loc[entry_idx] = entry_price
                apds.append(
                    mpf.make_addplot(
                        marker_series,
                        panel=0,
                        type="scatter",
                        marker=marker,
                        markersize=300,
                        color=color,
                    )
                )

                exit_time = trade.get("exit_time") or trade.get("close_time") or trade.get("end_time")
                if exit_time is not None:
                    exit_time = pd.Timestamp(exit_time)
                    if index_tz is not None:
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.tz_localize(index_tz)
                        else:
                            exit_time = exit_time.tz_convert(index_tz)
                    exit_idx_pos = data.index.get_indexer([exit_time], method="nearest")
                    end_idx = data.index[exit_idx_pos[0]] if exit_idx_pos[0] != -1 else data.index[-1]
                else:
                    end_idx = data.index[-1]

                if end_idx < entry_idx:
                    continue
                mask = (data.index >= entry_idx) & (data.index <= end_idx)
                if not mask.any():
                    continue

                def _add_band(lower_val, upper_val, fill_color, alpha=0.18):
                    if lower_val is None or upper_val is None:
                        return
                    low, high = sorted([lower_val, upper_val])
                    lower_series = pd.Series(index=data.index, dtype="float64")
                    upper_series = lower_series.copy()
                    lower_series.loc[mask] = low
                    upper_series.loc[mask] = high
                    apds.append(
                        mpf.make_addplot(
                            upper_series,
                            panel=0,
                            color=fill_color,
                            alpha=0,
                            fill_between=dict(y1=upper_series.values, y2=lower_series.values, alpha=alpha, color=fill_color),
                        )
                    )

                stop_loss = trade.get("stop_loss")
                take_profit = trade.get("take_profit")
                _add_band(entry_price, stop_loss, "#020202")
                _add_band(entry_price, take_profit, "#07A3F8")

        if len(zones) > 0:
            supply_zones = [z for z in zones if z.zone_type == "supply"]
            demand_zones = [z for z in zones if z.zone_type == "demand"]
            _zones_to_addplots(supply_zones, "tab:red")
            _zones_to_addplots(demand_zones, "tab:green")

        mpf.plot(
            data,
            type="candle",
            addplot=apds,
        )

        mpf.show()

if __name__ == "__main__":
    client = YFinanceClient()
    start_date = datetime.date(2024, 11, 10)
    end_date = datetime.date(2025, 11, 13)

    data_ = client.get_between(
        "USDCHF=X",
        start_date,
        end_date,
        interval="1wk",
    )
    client.plot_candlestick(data_)
    data_["close_open"] = data_["Close"] - data_["Open"]
    print(f"\nClose-Open ranges for USDCHF from 2024-11-10 to 2025-11-13:")
    print(data_[["Close", "close_open"]])
