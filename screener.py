"""Momentum day-trading screener (Warrior-Trading style).

Scans the entire US tradable universe and fires alerts on momentum movers using
a 3-stage funnel so it scales to thousands of symbols:

  Stage 1  BULK PRE-FILTER  - one Polygon full-market snapshot call filters
                              ~8,000 tickers down to a handful using cheap
                              criteria (price, % change, relative volume).
  Stage 2  ENRICH           - only survivors get the expensive per-symbol
                              Finnhub calls (news catalyst, float).
  Stage 3  REAL-TIME WATCH  - a Polygon websocket streams the survivors and
                              fires alerts the instant a threshold is crossed.

Criteria (all configurable in ``configs/screener.json``):
  1. Relative volume > 5x        (cheap - snapshot)
  2. Up > 30% on the day         (cheap - snapshot)
  3. Price between $3 and $20     (cheap - snapshot)
  4. News catalyst < 24h         (expensive - Finnhub)
  5. Float < 5,000,000 shares     (expensive - Finnhub, approximate)

Usage:
    export POLYGON_API_KEY=your_polygon_key
    export FINNHUB_API_KEY=your_finnhub_key
    python screener.py                 # one funnel pass
    python screener.py --loop          # repeat scan every scan_interval_sec
    python screener.py --watch         # funnel once, then live websocket alerts
"""

import os
import sys
import json
import time
import argparse
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.finnhub_client import FinnhubClient, FinnhubError
from utils.polygon_client import PolygonClient, PolygonWebSocket, PolygonError
from utils.notifier import build_notifiers, Notifier
from utils.storage import AlertStore, TOP_GAINERS, SMALL_CAP, LOW_FLOAT
from utils.momentum import momentum_score, MomentumScore


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class StockSnapshot:
    """A point-in-time view of a symbol used to evaluate criteria."""

    symbol: str
    price: Optional[float] = None              # current price ($)
    change_pct: Optional[float] = None         # day change (%)
    day_volume: Optional[float] = None         # today's cumulative volume
    avg_volume: Optional[float] = None         # average daily volume
    float_shares: Optional[float] = None       # free-floating shares (approx)
    news: List[Dict[str, Any]] = field(default_factory=list)
    company_name: Optional[str] = None
    prev_close: Optional[float] = None          # previous day close (for live %)
    security_type: Optional[str] = None         # Polygon type code (CS/ETF/ETN/...)

    @property
    def relative_volume(self) -> Optional[float]:
        if self.day_volume is None or not self.avg_volume:
            return None
        return self.day_volume / self.avg_volume


@dataclass
class CriterionResult:
    name: str
    passed: bool
    detail: str


# A criterion takes a snapshot + config and returns a result.
Criterion = Callable[["StockSnapshot", dict], CriterionResult]


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def relative_volume_criterion(snap: StockSnapshot, cfg: dict) -> CriterionResult:
    threshold = cfg.get("min_relative_volume", 5.0)
    rvol = snap.relative_volume
    if rvol is None:
        return CriterionResult("relative_volume", False, "rvol unavailable")
    return CriterionResult(
        "relative_volume", rvol >= threshold, f"{rvol:.1f}x (>= {threshold}x)"
    )


def change_pct_criterion(snap: StockSnapshot, cfg: dict) -> CriterionResult:
    threshold = cfg.get("min_change_pct", 30.0)
    if snap.change_pct is None:
        return CriterionResult("change_pct", False, "change unavailable")
    return CriterionResult(
        "change_pct", snap.change_pct >= threshold,
        f"{snap.change_pct:.1f}% (>= {threshold}%)",
    )


def price_range_criterion(snap: StockSnapshot, cfg: dict) -> CriterionResult:
    low = cfg.get("min_price", 3.0)
    high = cfg.get("max_price", 20.0)
    if snap.price is None:
        return CriterionResult("price_range", False, "price unavailable")
    ok = low <= snap.price <= high
    return CriterionResult("price_range", ok, f"${snap.price:.2f} (${low}-${high})")


def float_criterion(snap: StockSnapshot, cfg: dict) -> CriterionResult:
    max_float = cfg.get("max_float_shares", 5_000_000)
    if snap.float_shares is None:
        return CriterionResult("float", False, "float unavailable")
    ok = snap.float_shares <= max_float
    return CriterionResult(
        "float", ok, f"{snap.float_shares/1e6:.2f}M (<= {max_float/1e6:.1f}M)"
    )


def news_catalyst_criterion(snap: StockSnapshot, cfg: dict) -> CriterionResult:
    keywords = [k.lower() for k in cfg.get("catalyst_keywords", [])]
    max_age_hours = cfg.get("news_max_age_hours", 24)
    now = time.time()
    for item in snap.news:
        ts = item.get("datetime", 0)
        if ts and (now - ts) > max_age_hours * 3600:
            continue
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()
        hit = next((k for k in keywords if k in text), None)
        if hit:
            return CriterionResult("news_catalyst", True, f"'{hit}': {item.get('headline', '')[:60]}")
    return CriterionResult("news_catalyst", False, "no fresh catalyst")


# Each criterion is tagged "cheap" (evaluable from the bulk snapshot alone) or
# "expensive" (needs a per-symbol Finnhub call). The funnel runs cheap ones on
# the whole market, expensive ones only on survivors.
ALL_CRITERIA: Dict[str, Tuple[Criterion, str]] = {
    "relative_volume": (relative_volume_criterion, "cheap"),
    "change_pct": (change_pct_criterion, "cheap"),
    "price_range": (price_range_criterion, "cheap"),
    "float": (float_criterion, "expensive"),
    "news_catalyst": (news_catalyst_criterion, "expensive"),
}


def select_criteria(cfg: dict, stage: str) -> List[Criterion]:
    """Return enabled criterion functions for a funnel stage ("cheap"/"expensive")."""
    enabled = cfg.get("criteria", {})
    return [
        fn for name, (fn, kind) in ALL_CRITERIA.items()
        if kind == stage and enabled.get(name, True)
    ]


# ---------------------------------------------------------------------------
# Exclusion filters (post-enrichment)
# ---------------------------------------------------------------------------
# Polygon security "type" codes that are not common stock. ETFs/ETNs/funds are
# excluded by default so the screener focuses on individual momentum names.
_NON_STOCK_TYPES = {"ETF", "ETN", "ETV", "ETS", "FUND", "BASKET", "SP"}

# Name fragments that flag leveraged / inverse products (ProShares, Direxion,
# GraniteShares, etc.). Matched case-insensitively against the company name.
_DEFAULT_LEVERAGED_KEYWORDS = [
    "leveraged", "ultrapro", "ultra ", "ultrashort", "2x", "3x", "-1x",
    "1.5x", "2.25x", "daily bull", "daily bear", "bull ", "bear ",
    "long shares", "short shares", "inverse", "proshares", "direxion",
    "graniteshares", "microsectors",
]


def exclusion_reason(snap: StockSnapshot, cfg: dict) -> Optional[str]:
    """Return a reason string if ``snap`` should be filtered out, else ``None``.

    Configurable via ``configs/screener.json`` under a ``filters`` object::

        "filters": {
            "exclude_etfs": true,
            "exclude_leveraged": true,
            "leveraged_keywords": ["2x", "3x", ...],   # overrides defaults
            "exclude_name_keywords": ["acquisition"],  # extra name blocklist
            "exclude_symbols": ["SQQQ", "TQQQ"]        # explicit blocklist
        }
    """
    filters = cfg.get("filters", {})

    # 1. Explicit symbol blocklist.
    blocklist = {s.upper() for s in filters.get("exclude_symbols", [])}
    if snap.symbol and snap.symbol.upper() in blocklist:
        return "blocklisted symbol"

    # 2. Non-stock security types (ETF/ETN/fund) from Polygon ticker details.
    if filters.get("exclude_etfs", True) and snap.security_type:
        if snap.security_type.upper() in _NON_STOCK_TYPES:
            return f"security type {snap.security_type}"

    name = (snap.company_name or "").lower()

    # 3. Leveraged / inverse products by name.
    if filters.get("exclude_leveraged", True) and name:
        keywords = filters.get("leveraged_keywords") or _DEFAULT_LEVERAGED_KEYWORDS
        hit = next((k for k in keywords if k.lower() in name), None)
        if hit:
            return f"leveraged/inverse name ('{hit.strip()}')"

    # 4. Extra custom name keywords.
    extra = filters.get("exclude_name_keywords", [])
    if name and extra:
        hit = next((k for k in extra if k.lower() in name), None)
        if hit:
            return f"excluded keyword ('{hit}')"

    return None


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------
def categorize(snap: StockSnapshot, cfg: dict) -> List[str]:
    """Bucket a qualifying snapshot into Warrior-style daily lists.

    * ``top_gainers`` - main screener bucket: price in ``min_price``..``max_price``.
    * ``small_cap``   - second-tier bucket: price in ``small_cap_min_price``..<``small_cap_max_price``.
    * ``low_float``   - either of the above, plus float below ``low_float_max_shares``.
    """
    cats: List[str] = []
    main_min = cfg.get("min_price", 3.0)
    main_max = cfg.get("max_price", 20.0)
    small_cap_min = cfg.get("small_cap_min_price", 1.0)
    small_cap_max = cfg.get("small_cap_max_price", 3.0)
    low_float_max = cfg.get("low_float_max_shares", 5_000_000)

    if snap.price is not None:
        if main_min <= snap.price <= main_max:
            cats.append(TOP_GAINERS)
        elif small_cap_min <= snap.price < small_cap_max:
            cats.append(SMALL_CAP)

    if cats and snap.float_shares is not None and snap.float_shares < low_float_max:
        cats.append(LOW_FLOAT)
    return cats


# ---------------------------------------------------------------------------
# Momentum analysis (1-min technicals + score)
# ---------------------------------------------------------------------------
class MomentumAnalyzer:
    """Fetch a symbol's recent 1-min bars and compute its momentum score."""

    def __init__(self, polygon: PolygonClient, cfg: dict) -> None:
        self.polygon = polygon
        self.cfg = cfg
        self.lookback_days = cfg.get("momentum_lookback_days", 1)
        self.weights = cfg.get("momentum_weights")
        self.rvol_saturation = cfg.get("momentum_rvol_saturation", 10.0)

    def analyze(self, snap: StockSnapshot) -> Optional[MomentumScore]:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=self.lookback_days)
        try:
            bars = self.polygon.aggregates(
                snap.symbol, start, end,
                multiplier=1, timespan="minute",
                include_extended_hours=self.cfg.get("extended_hours", True),
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the scan
            print(f"[momentum] {snap.symbol} aggregates failed: {exc}")
            return None
        if bars is None or bars.empty:
            return None
        return momentum_score(
            bars, symbol=snap.symbol,
            weights=self.weights, rvol_saturation=self.rvol_saturation,
        )


# ---------------------------------------------------------------------------
# Stage 1: bulk snapshot pre-filter
# ---------------------------------------------------------------------------
class BulkPreFilter:
    """Filters the whole-market snapshot with the cheap criteria."""

    def __init__(self, polygon: PolygonClient, cfg: dict) -> None:
        self.polygon = polygon
        self.cfg = cfg
        self.cheap_criteria = select_criteria(cfg, "cheap")
        # Average volume isn't in the snapshot; approximate with previous-day
        # volume so relative volume can be evaluated cheaply in Stage 1.
        self.rvol_uses_prev_day = cfg.get("rvol_uses_prev_day_volume", True)

    def run(self) -> List[StockSnapshot]:
        rows = self.polygon.full_market_snapshot(
            include_otc=self.cfg.get("include_otc", False)
        )
        survivors: List[StockSnapshot] = []
        for row in rows:
            symbol = row.get("ticker")
            if not symbol:
                continue
            snap = StockSnapshot(
                symbol=symbol,
                price=row.get("price"),
                change_pct=row.get("change_pct"),
                day_volume=row.get("day_volume"),
                avg_volume=row.get("prev_volume") if self.rvol_uses_prev_day else None,
                prev_close=row.get("prev_close"),
            )
            if all(fn(snap, self.cfg).passed for fn in self.cheap_criteria):
                survivors.append(snap)
        # Second pass: small cap price range ($1-$3) with its own price override.
        # Reuses the same rvol/change criteria; only min_price/max_price differ.
        small_cap_min = self.cfg.get("small_cap_min_price", 1.0)
        small_cap_max = self.cfg.get("small_cap_max_price", 3.0)
        if small_cap_min < self.cfg.get("min_price", 3.0):
            small_cap_cfg = dict(self.cfg, min_price=small_cap_min, max_price=small_cap_max)
            seen = {s.symbol for s in survivors}
            for row in rows:
                symbol = row.get("ticker")
                if not symbol or symbol in seen:
                    continue
                snap = StockSnapshot(
                    symbol=symbol,
                    price=row.get("price"),
                    change_pct=row.get("change_pct"),
                    day_volume=row.get("day_volume"),
                    avg_volume=row.get("prev_volume") if self.rvol_uses_prev_day else None,
                    prev_close=row.get("prev_close"),
                )
                if all(fn(snap, small_cap_cfg).passed for fn in self.cheap_criteria):
                    survivors.append(snap)
        return survivors


# ---------------------------------------------------------------------------
# Stage 2: per-symbol enrichment (Finnhub)
# ---------------------------------------------------------------------------
class Enricher:
    """Adds news + float (and a better average volume) to survivor snapshots."""

    def __init__(self, finnhub: FinnhubClient, cfg: dict) -> None:
        self.finnhub = finnhub
        self.cfg = cfg
        self.expensive_criteria = select_criteria(cfg, "expensive")
        # Float is not required for the base Top Gainers / Small Cap lists.
        # It is only used to decide whether a base hit also belongs in Low Float.
        self.base_criteria = [fn for fn in self.expensive_criteria if fn is not float_criterion]

    def enrich(self, snap: StockSnapshot) -> StockSnapshot:
        profile = self._safe(self.finnhub.company_profile, snap.symbol) or {}
        snap.company_name = profile.get("name")
        shares_out_m = profile.get("shareOutstanding")  # millions

        metrics = (self._safe(self.finnhub.basic_financials, snap.symbol) or {}).get("metric", {})
        # NOTE: free-float is approximate - prefer an explicit float metric,
        # otherwise fall back to shares outstanding.
        float_m = metrics.get("floatShares") or metrics.get("sharesOutstanding") or shares_out_m
        if float_m is not None:
            snap.float_shares = float(float_m) * 1e6

        # Prefer a real average volume for relative-volume accuracy.
        avg_vol_m = (metrics.get("10DayAverageTradingVolume")
                     or metrics.get("3MonthAverageTradingVolume"))
        if avg_vol_m:
            snap.avg_volume = float(avg_vol_m) * 1e6

        snap.news = self._safe(self.finnhub.company_news, snap.symbol) or []
        return snap

    def evaluate(self, snap: StockSnapshot) -> Tuple[bool, List[CriterionResult]]:
        results = [fn(snap, self.cfg) for fn in self.base_criteria]
        return all(r.passed for r in results), results

    @staticmethod
    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the scan
            print(f"[enricher] {fn.__name__}({args[0]}) failed: {exc}")
            return None


# ---------------------------------------------------------------------------
# Alerts (shared by funnel + websocket)
# ---------------------------------------------------------------------------
class AlertManager:
    def __init__(self, notifiers: List[Notifier], cooldown_sec: int) -> None:
        self.notifiers = notifiers
        self.cooldown_sec = cooldown_sec
        self._alerted: Dict[str, float] = {}

    def alert(self, snap: StockSnapshot, results: List[CriterionResult],
              prefix: str = "🚀") -> bool:
        now = time.time()
        if now - self._alerted.get(snap.symbol, 0) < self.cooldown_sec:
            return False  # cooldown active; don't spam
        self._alerted[snap.symbol] = now

        title = f"{prefix} {snap.symbol} momentum alert"
        lines = [
            f"{snap.company_name or snap.symbol} ({snap.symbol})",
            f"Price: ${snap.price:.2f}" if snap.price else "Price: n/a",
            f"Change: {snap.change_pct:.1f}%" if snap.change_pct is not None else "",
        ]
        lines += [f"- {r.name}: {r.detail}" for r in results]
        message = "\n".join(l for l in lines if l)
        for notifier in self.notifiers:
            try:
                notifier.send(title, message)
            except Exception as exc:  # noqa: BLE001
                print(f"[alert] {notifier.name} failed: {exc}")
        return True


# ---------------------------------------------------------------------------
# Stage 3: real-time websocket watcher
# ---------------------------------------------------------------------------
class WebSocketWatcher:
    """Streams survivor symbols and re-checks the cheap criteria live."""

    def __init__(self, cfg: dict, alerts: AlertManager,
                 api_key: Optional[str] = None) -> None:
        self.cfg = cfg
        self.alerts = alerts
        self.cheap_criteria = select_criteria(cfg, "cheap")
        self.ws = PolygonWebSocket(
            api_key=api_key, channel=cfg.get("ws_channel", "AM")
        )
        # symbol -> enriched snapshot captured at funnel time
        self._watched: Dict[str, StockSnapshot] = {}

    def watch(self, snapshots: List[StockSnapshot]) -> None:
        self._watched = {s.symbol: s for s in snapshots}
        symbols = list(self._watched.keys())
        if not symbols:
            print("[ws] no symbols to watch")
            return
        print(f"[ws] watching {len(symbols)} symbol(s) live: {', '.join(symbols)}")
        self.ws.run(symbols, self._on_update)

    def update(self, snapshots: List[StockSnapshot]) -> None:
        self._watched = {s.symbol: s for s in snapshots}
        self.ws.update_symbols(list(self._watched.keys()))

    def _on_update(self, symbol: str, payload: Dict[str, Any]) -> None:
        snap = self._watched.get(symbol)
        if snap is None:
            return
        price = payload.get("price")
        if price:
            snap.price = price
            if snap.prev_close:
                snap.change_pct = (price - snap.prev_close) / snap.prev_close * 100.0
        acc_vol = payload.get("accumulated_volume")
        if acc_vol:
            snap.day_volume = acc_vol

        results = [fn(snap, self.cfg) for fn in self.cheap_criteria]
        if all(r.passed for r in results):
            self.alerts.alert(snap, results, prefix="⚡")

    def stop(self) -> None:
        self.ws.stop()


# ---------------------------------------------------------------------------
# Screener orchestrator
# ---------------------------------------------------------------------------
class Screener:
    def __init__(self, cfg: dict, *, polygon: PolygonClient,
                 finnhub: Optional[FinnhubClient] = None,
                 notifiers: Optional[List[Notifier]] = None,
                 store: Optional[AlertStore] = None) -> None:
        self.cfg = cfg
        self.polygon = polygon
        self.finnhub = finnhub
        self.prefilter = BulkPreFilter(polygon, cfg)
        self.enricher = Enricher(finnhub, cfg) if finnhub else None
        self.alerts = AlertManager(
            notifiers or build_notifiers(cfg.get("notifiers", [])),
            cfg.get("alert_cooldown_sec", 3600),
        )
        self.store = store or AlertStore(cfg.get("data_dir", "data"))
        self.momentum = (
            MomentumAnalyzer(polygon, cfg)
            if cfg.get("compute_momentum", True) else None
        )

    def _enrich_float_polygon(self, hits: List[StockSnapshot]) -> None:
        """Fill missing float / company name from Polygon ticker details.

        Polygon exposes ``share_class_shares_outstanding`` (used as the float
        proxy) via ``/v3/reference/tickers/{ticker}``. Only fetched for symbols
        still missing the data, so Finnhub values (if any) take precedence.
        """
        for snap in hits:
            if (snap.float_shares is not None and snap.company_name
                    and snap.security_type is not None):
                continue
            try:
                details = self.polygon.ticker_details(snap.symbol)
            except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't stop the scan
                print(f"[polygon-float] {snap.symbol} failed: {exc}")
                continue
            if not snap.company_name:
                snap.company_name = details.get("company_name")
            if snap.security_type is None:
                snap.security_type = details.get("type")
            if snap.float_shares is None:
                shares = (details.get("share_class_shares_outstanding")
                          or details.get("weighted_shares_outstanding"))
                if shares:
                    snap.float_shares = float(shares)

    def _apply_filters(self, hits: List[StockSnapshot]) -> List[StockSnapshot]:
        """Drop hits matching the exclusion filters (e.g. leveraged ETFs)."""
        kept: List[StockSnapshot] = []
        for snap in hits:
            reason = exclusion_reason(snap, self.cfg)
            if reason:
                print(f"  Filter : {snap.symbol:<6} skipped ({reason})")
                continue
            kept.append(snap)
        return kept

    def _persist(self, hits: List[StockSnapshot],
                 scores: Dict[str, Optional[MomentumScore]]) -> None:
        """Write the day's categorized lists and per-symbol alert records."""
        categorized: Dict[str, List[Dict[str, Any]]] = {
            TOP_GAINERS: [], SMALL_CAP: [], LOW_FLOAT: [],
        }
        for snap in hits:
            cats = categorize(snap, self.cfg)
            score = scores.get(snap.symbol)
            # Preserve the symbol's first trigger time across re-scans.
            triggered_at = (self.store.first_triggered_at(snap.symbol)
                            or datetime.datetime.now().isoformat(timespec="seconds"))
            row = {
                "symbol": snap.symbol,
                "company_name": snap.company_name,
                "price": snap.price,
                "change_pct": snap.change_pct,
                "relative_volume": snap.relative_volume,
                "float_shares": snap.float_shares,
                "categories": cats,
                "triggered_at": triggered_at,
                "momentum_score": score.score if score else None,
                "momentum": score.as_dict() if score else None,
                "headline": snap.news[0].get("headline") if snap.news else None,
            }
            for cat in cats:
                categorized[cat].append(row)
            self.store.record_alert(row)
        self.store.save_lists(categorized)

    def scan(self) -> List[StockSnapshot]:
        """Run Stage 1 + Stage 2; alert, score, persist + return hits."""
        survivors = self.prefilter.run()
        print(f"  Stage 1: {len(survivors)} survivor(s) after cheap filters")

        if not self.enricher or not select_criteria(self.cfg, "expensive"):
            for snap in survivors:
                self.alerts.alert(snap, [], prefix="🚀")
            self._enrich_float_polygon(survivors)
            survivors = self._apply_filters(survivors)
            scores = self._score_hits(survivors)
            self._persist(survivors, scores)
            return survivors

        hits: List[StockSnapshot] = []
        for snap in survivors:
            self.enricher.enrich(snap)
            passed, results = self.enricher.evaluate(snap)
            tag = "HIT" if passed else "   "
            summary = ", ".join(f"{r.name}={'Y' if r.passed else 'N'}" for r in results)
            print(f"  Stage 2: {snap.symbol:<6} {tag} {summary}")
            if passed:
                hits.append(snap)
                self.alerts.alert(snap, results)
        print(f"  Stage 2: {len(hits)} confirmed hit(s)")

        self._enrich_float_polygon(hits)
        hits = self._apply_filters(hits)
        scores = self._score_hits(hits)
        self._persist(hits, scores)
        return hits

    def _score_hits(self, hits: List[StockSnapshot]) -> Dict[str, Optional[MomentumScore]]:
        scores: Dict[str, Optional[MomentumScore]] = {}
        if not self.momentum:
            return scores
        for snap in hits:
            score = self.momentum.analyze(snap)
            scores[snap.symbol] = score
            if score is not None:
                print(f"  Momentum: {snap.symbol:<6} score={score.score:.0f}")
        return scores

    def run_loop(self) -> None:
        interval = self.cfg.get("scan_interval_sec", 60)
        print(f"Starting screener loop (every {interval}s). Ctrl-C to stop.")
        while True:
            stamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{stamp}] scanning full market...")
            self.scan()
            time.sleep(interval)

    def run_watch(self) -> None:
        """Funnel once, then keep the survivors under live websocket watch."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{stamp}] initial funnel scan...")
        hits = self.scan()
        watcher = WebSocketWatcher(
            self.cfg, self.alerts, api_key=self.polygon.api_key
        )
        watcher.watch(hits)
        rescan = self.cfg.get("watch_rescan_sec", 300)
        try:
            while True:
                time.sleep(rescan)
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"\n[{stamp}] re-scanning to refresh watch list...")
                hits = self.scan()
                watcher.update(hits)
        except KeyboardInterrupt:
            print("\nStopping live watch...")
            watcher.stop()


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "screener.json"
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Momentum day-trading screener")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to screener config JSON")
    parser.add_argument("--loop", action="store_true",
                        help="Repeat the funnel scan every scan_interval_sec")
    parser.add_argument("--watch", action="store_true",
                        help="Funnel once, then stream live websocket alerts")
    parser.add_argument("--polygon-key", default=None,
                        help="Polygon API key (overrides POLYGON_API_KEY)")
    parser.add_argument("--finnhub-key", default=None,
                        help="Finnhub API key (overrides FINNHUB_API_KEY)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    polygon_key = (args.polygon_key or cfg.get("polygon_api_key")
                   or os.environ.get("POLYGON_API_KEY"))
    try:
        polygon = PolygonClient(api_key=polygon_key)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    finnhub_key = (args.finnhub_key or cfg.get("finnhub_api_key")
                   or os.environ.get("FINNHUB_API_KEY"))
    finnhub: Optional[FinnhubClient] = None
    if finnhub_key:
        try:
            finnhub = FinnhubClient(
                api_key=finnhub_key,
                min_interval=cfg.get("min_request_interval_sec", 1.05),
            )
        except ValueError as exc:
            print(f"Warning: Finnhub disabled ({exc}); running cheap criteria only.")
    else:
        print("Warning: no Finnhub key; running cheap (snapshot) criteria only.")

    screener = Screener(cfg, polygon=polygon, finnhub=finnhub)

    if args.watch:
        screener.run_watch()
    elif args.loop:
        try:
            screener.run_loop()
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        hits = screener.scan()
        print(f"\nScan complete: {len(hits)} hit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
