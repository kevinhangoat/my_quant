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
        zones = []
    ):
        """Plot a candlestick chart with optional supply/demand zones."""

        apds = []

        def _zones_to_addplots(zones, color):
            for zone in zones:
                start_time, low, high = zone.end, zone.lower, zone.upper
                start_time = pd.Timestamp(start_time)
                index_tz = getattr(data.index, "tz", None)
                if index_tz is not None:
                    if start_time.tzinfo is None:
                        start_time = start_time.tz_localize(index_tz)
                    else:
                        start_time = start_time.tz_convert(index_tz)
                mask = data.index >= start_time
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
    start_date = datetime.date(2025, 11, 10)
    end_date = datetime.date(2025, 11, 13)

    data_ = client.get_between(
        "CL=F",
        start_date,
        end_date,
        interval="1h",
    )
    client.plot_candlestick(
        data_,
        supply_zones=[(datetime.datetime(2025, 11, 11, 0), 60, 60.5)],
        demand_zones=[(datetime.datetime(2025, 11, 11, 3), 54, 53.5)],
    )

    data_["close_open"] = data_["Close"] - data_["Open"]
    print(f"\nClose-Open ranges for CL=F from 2025-11-10 to 2025-11-13:")
    print(data_[["Close", "close_open"]])
