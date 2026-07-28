"""Persistent storage for screener alerts and daily watch lists.

Everything is written as plain JSON, one directory per trading day::

    data/
      2026-06-28/
        alerts.json        # every alert fired that day (deduped per cooldown)
        top_gainers.json   # symbols passing the core 4 criteria
        small_cap.json     # passing symbols priced < small_cap_max_price
        low_float.json     # passing symbols with float < low_float_max_shares

The format is intentionally simple so the web dashboard, the backtest reports
and ad-hoc analysis can all read it without a database. Writes are atomic
(write-to-temp then ``os.replace``) so a reader never sees a half-written file.
"""

import json
import os
import tempfile
import datetime
from typing import Any, Dict, List, Optional

# Categories tracked as standalone daily lists.
TOP_GAINERS = "top_gainers"
SMALL_CAP = "small_cap"
LOW_FLOAT = "low_float"
CATEGORIES = (TOP_GAINERS, SMALL_CAP, LOW_FLOAT)


def _today_str() -> str:
    return datetime.date.today().isoformat()


class AlertStore:
    """Read/write screener output as dated JSON files under ``root``."""

    def __init__(self, root: str = "data") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _day_dir(self, date: Optional[str] = None) -> str:
        path = os.path.join(self.root, date or _today_str())
        os.makedirs(path, exist_ok=True)
        return path

    def _file(self, name: str, date: Optional[str] = None) -> str:
        return os.path.join(self._day_dir(date), f"{name}.json")

    # ------------------------------------------------------------------
    # Low-level atomic JSON IO
    # ------------------------------------------------------------------
    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            return default

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        directory = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def record_alert(self, alert: Dict[str, Any], *,
                     date: Optional[str] = None, dedupe: bool = True) -> bool:
        """Append an alert to the day's ``alerts.json``.

        ``alert`` should at least carry ``symbol``; a ``timestamp`` (epoch
        seconds) is added if missing. When ``dedupe`` is True an existing alert
        for the same symbol is replaced rather than duplicated, so category
        changes (e.g. top_gainers -> small_cap after rule updates) do not leave
        stale rows behind. Returns True if a new symbol was added (vs. updating
        an existing one).
        """
        alert = dict(alert)
        alert.setdefault("timestamp", datetime.datetime.now().timestamp())
        alert.setdefault("time", datetime.datetime.now().isoformat(timespec="seconds"))

        path = self._file("alerts", date)
        alerts: List[Dict[str, Any]] = self._read_json(path, [])

        is_new = True
        if dedupe:
            key = alert.get("symbol")
            for i, existing in enumerate(alerts):
                existing_key = existing.get("symbol")
                if existing_key == key:
                    alerts[i] = alert
                    is_new = False
                    break
        if is_new:
            alerts.append(alert)

        self._write_json(path, alerts)
        return is_new

    def load_alerts(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._read_json(self._file("alerts", date), [])

    def first_triggered_at(self, symbol: str, date: Optional[str] = None) -> Optional[str]:
        """Return the earliest recorded trigger time for ``symbol`` today.

        Used to keep a stock's *first* trigger time stable across re-scans
        instead of overwriting it on every funnel pass.
        """
        times = [
            a.get("triggered_at") or a.get("time")
            for a in self.load_alerts(date)
            if a.get("symbol") == symbol and (a.get("triggered_at") or a.get("time"))
        ]
        return min(times) if times else None

    # ------------------------------------------------------------------
    # Daily category lists
    # ------------------------------------------------------------------
    def save_list(self, category: str, rows: List[Dict[str, Any]], *,
                  date: Optional[str] = None) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Use one of {CATEGORIES}.")
        self._write_json(self._file(category, date), rows)

    def save_lists(self, categorized: Dict[str, List[Dict[str, Any]]], *,
                   date: Optional[str] = None) -> None:
        for category in CATEGORIES:
            self.save_list(category, categorized.get(category, []), date=date)

    def load_list(self, category: str, date: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._read_json(self._file(category, date), [])

    def load_day(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Return all stored data for a day: alerts + each category list."""
        return {
            "date": date or _today_str(),
            "alerts": self.load_alerts(date),
            **{cat: self.load_list(cat, date) for cat in CATEGORIES},
        }

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_days(self) -> List[str]:
        """Return stored trading-day folders, newest first."""
        if not os.path.isdir(self.root):
            return []
        days = [
            name for name in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, name)) and _looks_like_date(name)
        ]
        return sorted(days, reverse=True)


def _looks_like_date(name: str) -> bool:
    try:
        datetime.date.fromisoformat(name)
        return True
    except ValueError:
        return False
