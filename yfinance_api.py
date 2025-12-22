import datetime
from typing import Optional
import yfinance as yf
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

if __name__ == "__main__":
    client = YFinanceClient()
    data_1d = client.get_past_three_years("AAPL")
    data_4h = client.get_past_three_months("AAPL")
    print(data_1d.tail(20))
    print(data_4h.tail(20))