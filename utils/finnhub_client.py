"""Lightweight Finnhub REST API client.

Only depends on the ``requests`` package and the standard library so it runs
anywhere the rest of this project does (no extra SDK required).

Get a free API key at https://finnhub.io/ and expose it via the
``FINNHUB_API_KEY`` environment variable, or pass ``api_key`` explicitly.

Notes on the free tier:
* ``/quote``, ``/stock/profile2``, ``/stock/metric`` and ``/company-news``
  are available on the free plan.
* ``/stock/candle`` (intraday OHLCV) requires a paid plan for US stocks.
  The screener degrades gracefully when candle data is unavailable.
"""

import os
import time
import datetime
from typing import Any, Dict, List, Optional

import requests


class FinnhubError(RuntimeError):
    """Raised when the Finnhub API returns an error response."""


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = 10.0,
        min_interval: float = 1.05,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Finnhub API key missing. Pass api_key=... or set FINNHUB_API_KEY."
            )
        self.timeout = timeout
        # Free tier allows ~60 calls/min; throttle to stay under the limit.
        self.min_interval = min_interval
        self._last_call = 0.0
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Core request helper
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        params["token"] = self.api_key

        # Simple client-side rate limiting.
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        self._last_call = time.monotonic()

        if resp.status_code == 429:
            raise FinnhubError("Rate limit exceeded (HTTP 429). Slow down requests.")
        if resp.status_code == 403:
            raise FinnhubError(
                f"Access denied for {path} (HTTP 403). This endpoint may "
                "require a paid Finnhub plan."
            )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    def quote(self, symbol: str) -> Dict[str, Any]:
        """Real-time quote.

        Returns keys: ``c`` (current), ``d`` (change), ``dp`` (percent change),
        ``h`` (high), ``l`` (low), ``o`` (open), ``pc`` (previous close).
        """
        return self._get("quote", {"symbol": symbol})

    def company_profile(self, symbol: str) -> Dict[str, Any]:
        """Company profile (name, exchange, ``shareOutstanding`` in millions)."""
        return self._get("stock/profile2", {"symbol": symbol})

    def basic_financials(self, symbol: str, metric: str = "all") -> Dict[str, Any]:
        """Basic financial metrics, including average trading volumes."""
        return self._get("stock/metric", {"symbol": symbol, "metric": metric})

    def company_news(
        self,
        symbol: str,
        *,
        from_date: Optional[datetime.date] = None,
        to_date: Optional[datetime.date] = None,
    ) -> List[Dict[str, Any]]:
        """Company news between two dates (defaults to the last 2 days)."""
        today = datetime.date.today()
        from_date = from_date or (today - datetime.timedelta(days=2))
        to_date = to_date or today
        return self._get(
            "company-news",
            {
                "symbol": symbol,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
            },
        )

    def stock_candles(
        self,
        symbol: str,
        resolution: str,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Dict[str, Any]:
        """Intraday/daily OHLCV candles. Requires a paid plan for US stocks.

        ``resolution`` is one of ``1, 5, 15, 30, 60, D, W, M``.
        """
        return self._get(
            "stock/candle",
            {
                "symbol": symbol,
                "resolution": resolution,
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            },
        )

    def us_symbols(self, exchange: str = "US") -> List[Dict[str, Any]]:
        """List all tradable symbols on an exchange (default US)."""
        return self._get("stock/symbol", {"exchange": exchange})
