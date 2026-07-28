"""Polygon.io client: full-market snapshot (REST) + real-time stream (websocket).

Two pieces power the screener funnel:

* :class:`PolygonClient` - one REST call returns a snapshot of *every* US ticker
  (price, % change, today's volume, previous-day volume). This is Stage 1 of the
  funnel: filter ~8,000 symbols down to a handful locally, for the cost of a
  single request.
* :class:`PolygonWebSocket` - subscribes to per-minute aggregates for a small set
  of surviving symbols and invokes a callback on every update. This is Stage 3:
  real-time alerts without polling.

Only depends on ``requests`` and ``websocket-client``.

Get an API key at https://polygon.io/ and expose it via ``POLYGON_API_KEY`` or
pass ``api_key`` explicitly.
"""

import os
import json
import datetime
import threading
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    import pandas as pd
except ImportError:  # pragma: no cover - only needed for aggregate frames
    pd = None

try:
    import websocket  # websocket-client
except ImportError:  # pragma: no cover - only needed for the live stream
    websocket = None


class PolygonError(RuntimeError):
    """Raised when the Polygon REST API returns an error response."""


EASTERN_TZ = "America/New_York"   # US trading-session reference (day boundaries)
DISPLAY_TZ = "Europe/Zurich"      # timezone used for displayed/index timestamps


def _to_date_str(value: "datetime.date | datetime.datetime | str") -> str:
    """Coerce a date/datetime/str into Polygon's ``YYYY-MM-DD`` path format."""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    raise TypeError(f"Unsupported date value: {value!r}")


def _aggregates_to_frame(results: List[Dict[str, Any]], include_extended_hours: bool):
    """Convert raw Polygon aggregate rows into an ET-indexed OHLCV DataFrame."""
    if pd is None:
        raise ImportError("pandas is required for aggregates(). Install pandas.")
    if not results:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]
        )
    frame = pd.DataFrame(results)
    frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").tz_convert(DISPLAY_TZ)
    frame = frame.rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close",
        "v": "Volume", "vw": "VWAP", "n": "Transactions",
    })
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions")
            if c in frame.columns]
    frame = frame[keep]
    if not include_extended_hours:
        frame = frame.between_time("09:30", "16:00")
    return frame


class PolygonClient:
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 30.0,
                 session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Polygon API key missing. Pass api_key=... or set POLYGON_API_KEY."
            )
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 429:
            raise PolygonError("Rate limit exceeded (HTTP 429). Slow down requests.")
        if resp.status_code in (401, 403):
            raise PolygonError(
                f"Access denied for {path} (HTTP {resp.status_code}). Check your "
                "API key and plan."
            )
        resp.raise_for_status()
        return resp.json()

    def full_market_snapshot(
        self, include_otc: bool = False
    ) -> List[Dict[str, Any]]:
        """Return a snapshot of every US stock ticker in a single call.

        Each entry exposes (flattened from Polygon's nested shape):
          ``ticker``, ``price``, ``change_pct``, ``day_volume``,
          ``prev_volume``, ``prev_close``.
        """
        data = self._get(
            "v2/snapshot/locale/us/markets/stocks/tickers",
            {"include_otc": str(include_otc).lower()},
        )
        if data.get("status") not in ("OK", "DELAYED"):
            raise PolygonError(f"Snapshot failed: {data.get('status')}")
        out = []
        for t in data.get("tickers", []):
            day = t.get("day") or {}
            prev = t.get("prevDay") or {}
            last_trade = t.get("lastTrade") or {}
            minute = t.get("min") or {}
            # During the session ``day`` may be empty until the first trade;
            # fall back to the last trade / minute bar for a live price.
            price = day.get("c") or last_trade.get("p") or minute.get("c")
            # ``min.av`` is the accumulated volume for the day (live);
            # ``day.v`` is the official daily volume.
            day_volume = day.get("v") or minute.get("av")
            out.append({
                "ticker": t.get("ticker"),
                "price": price,
                "change_pct": t.get("todaysChangePerc"),
                "day_volume": day_volume,
                "prev_volume": prev.get("v"),
                "prev_close": prev.get("c"),
            })
        return out

    def ticker_details(self, ticker: str) -> Dict[str, Any]:
        """Return reference details for a ticker (name + shares outstanding).

        Wraps Polygon's ``/v3/reference/tickers/{ticker}`` endpoint. The result
        exposes ``company_name``, ``type`` (Polygon security type code, e.g.
        ``CS``/``ETF``/``ETN``), ``share_class_shares_outstanding`` (the float
        proxy) and ``weighted_shares_outstanding``; values are ``None`` when
        Polygon does not provide them.
        """
        data = self._get(f"v3/reference/tickers/{ticker}")
        if data.get("status") not in ("OK", "DELAYED"):
            raise PolygonError(f"Ticker details failed: {data.get('status')}")
        res = data.get("results") or {}
        return {
            "company_name": res.get("name"),
            "type": res.get("type"),
            "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
            "weighted_shares_outstanding": res.get("weighted_shares_outstanding"),
        }

    def grouped_daily(
        self,
        date: "datetime.date | str",
        *,
        adjusted: bool = True,
        include_otc: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Return daily OHLCV for *every* US ticker on ``date`` in one call.

        Wraps Polygon's ``/v2/aggs/grouped`` endpoint - the historical analogue
        of :meth:`full_market_snapshot`. Returns a ``{ticker: row}`` mapping
        where each row exposes ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``vwap`` and ``transactions``. An empty dict is returned for
        non-trading days (weekends/holidays).
        """
        date_str = _to_date_str(date)
        data = self._get(
            f"v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            {"adjusted": str(adjusted).lower(), "include_otc": str(include_otc).lower()},
        )
        if data.get("status") not in ("OK", "DELAYED"):
            raise PolygonError(f"Grouped daily failed: {data.get('status')}")
        out: Dict[str, Dict[str, Any]] = {}
        for row in data.get("results") or []:
            ticker = row.get("T")
            if not ticker:
                continue
            out[ticker] = {
                "open": row.get("o"),
                "high": row.get("h"),
                "low": row.get("l"),
                "close": row.get("c"),
                "volume": row.get("v"),
                "vwap": row.get("vw"),
                "transactions": row.get("n"),
            }
        return out

    def aggregates(
        self,
        ticker: str,
        start: "datetime.date | datetime.datetime | str",
        end: "datetime.date | datetime.datetime | str",
        *,
        multiplier: int = 1,
        timespan: str = "minute",
        adjusted: bool = True,
        include_extended_hours: bool = True,
        limit: int = 50000,
    ) -> "pd.DataFrame":
        """Return OHLCV aggregate bars for a ticker as a tz-aware DataFrame.

        Wraps Polygon's ``/v2/aggs`` endpoint (paginating with ``next_url``).
        ``start``/``end`` accept ``date``, ``datetime`` or ``YYYY-MM-DD`` strings.
        The returned frame is indexed by Europe/Zurich timestamps with capitalized
        columns (``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
        ``VWAP``, ``Transactions``) so it matches the existing plotting code.

        ``include_extended_hours`` keeps pre/post-market bars (Polygon returns
        them by default); set False to clip to 09:30-16:00 ET.
        """
        from_str = _to_date_str(start)
        to_str = _to_date_str(end)
        path = (
            f"v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/"
            f"{from_str}/{to_str}"
        )
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
        }
        results: List[Dict[str, Any]] = []
        data = self._get(path, params)
        results.extend(data.get("results") or [])
        # Follow pagination links until exhausted.
        next_url = data.get("next_url")
        while next_url:
            data = self._get_url(next_url)
            results.extend(data.get("results") or [])
            next_url = data.get("next_url")

        return _aggregates_to_frame(results, include_extended_hours)

    def _get_url(self, url: str) -> Any:
        """GET a fully-qualified Polygon URL (used to follow ``next_url``)."""
        sep = "&" if "?" in url else "?"
        resp = self.session.get(f"{url}{sep}apiKey={self.api_key}", timeout=self.timeout)
        if resp.status_code == 429:
            raise PolygonError("Rate limit exceeded (HTTP 429). Slow down requests.")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Real-time websocket stream
# ---------------------------------------------------------------------------
class PolygonWebSocket:
    """Subscribe to Polygon per-minute aggregates for a set of symbols.

    On every aggregate message the ``on_update(symbol, payload)`` callback is
    invoked with a normalized dict::

        {"symbol": "AAPL", "price": 8.42, "volume": 1234.0,
         "accumulated_volume": 5_000_000.0}

    Runs the socket on a background thread; call :meth:`stop` to shut down.
    """

    STREAM_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: Optional[str] = None, *,
                 channel: str = "AM", auto_reconnect: bool = True) -> None:
        if websocket is None:
            raise ImportError(
                "websocket-client is required for the live stream. "
                "Install it with: pip install websocket-client"
            )
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key missing for websocket stream.")
        self.channel = channel  # "AM" minute bars, "A" second bars, "T" trades
        self.auto_reconnect = auto_reconnect
        self._symbols: List[str] = []
        self._on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._ws: Optional["websocket.WebSocketApp"] = None
        self._thread: Optional[threading.Thread] = None
        self._authed = False
        self._stopped = False

    def run(self, symbols: List[str],
            on_update: Callable[[str, Dict[str, Any]], None]) -> None:
        """Start streaming (non-blocking). Updates flow to ``on_update``."""
        self._symbols = [s.upper() for s in symbols]
        self._on_update = on_update
        self._stopped = False
        self._connect()

    def _connect(self) -> None:
        self._ws = websocket.WebSocketApp(
            self.STREAM_URL,
            on_open=self._handle_open,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()

    def _handle_open(self, ws) -> None:
        ws.send(json.dumps({"action": "auth", "params": self.api_key}))

    def _subscribe(self, ws) -> None:
        if not self._symbols:
            return
        params = ",".join(f"{self.channel}.{s}" for s in self._symbols)
        ws.send(json.dumps({"action": "subscribe", "params": params}))

    def _handle_message(self, ws, raw) -> None:
        try:
            events = json.loads(raw)
        except (ValueError, TypeError):
            return
        for ev in events:
            etype = ev.get("ev")
            if etype == "status":
                if ev.get("status") == "auth_success":
                    self._authed = True
                    self._subscribe(ws)
                continue
            if etype != self.channel:
                continue
            symbol = ev.get("sym")
            payload = {
                "symbol": symbol,
                "price": ev.get("c"),                 # close of the bar
                "volume": ev.get("v"),                # bar volume
                "accumulated_volume": ev.get("av"),   # day's accumulated volume
            }
            if self._on_update and symbol:
                self._on_update(symbol, payload)

    def _handle_error(self, ws, error) -> None:
        print(f"[polygon-ws] error: {error}")

    def _handle_close(self, ws, *args) -> None:
        self._authed = False
        if self.auto_reconnect and not self._stopped:
            print("[polygon-ws] connection closed, reconnecting...")
            self._connect()

    def update_symbols(self, symbols: List[str]) -> None:
        """Re-subscribe to a new symbol set (e.g. after a fresh scan)."""
        self._symbols = [s.upper() for s in symbols]
        if self._ws and self._authed:
            self._subscribe(self._ws)

    def stop(self) -> None:
        self._stopped = True
        if self._ws is not None:
            self._ws.close()
