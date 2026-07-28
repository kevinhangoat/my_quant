"""FastAPI dashboard for the momentum screener.

Serves the cached daily lists (top gainers / small cap / low float) and per-
symbol momentum scores produced by ``screener.py`` and stored as JSON by
``utils.storage.AlertStore``. A small static frontend (``static/index.html``)
polls the JSON API and renders a Warrior-Trading-style table that auto-refreshes.

Run it::

    pip install fastapi uvicorn
    uvicorn web_app.app:app --reload --port 8000
    # then open http://127.0.0.1:8000

Optionally trigger a fresh scan from the UI (POST /api/scan) if Polygon/Finnhub
keys are configured in ``configs/screener.json`` or the environment.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.storage import AlertStore, CATEGORIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DATA_DIR = os.environ.get("SCREENER_DATA_DIR", os.path.join(ROOT, "data"))

app = FastAPI(title="Momentum Screener", version="1.0")
store = AlertStore(DATA_DIR)


@app.get("/api/days", response_model=List[str])
def list_days() -> List[str]:
    """Return stored trading days, newest first."""
    return store.list_days()


@app.get("/api/lists")
def get_lists(date: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """Return all categorized lists + alerts for a day (defaults to latest)."""
    if date is None:
        days = store.list_days()
        date = days[0] if days else None
    if date is None:
        return {"date": None, "alerts": [], **{c: [] for c in CATEGORIES}}
    return store.load_day(date)


@app.post("/api/scan")
def trigger_scan() -> Dict[str, Any]:
    """Run one screener funnel pass on demand. Requires API keys configured."""
    try:
        from screener import Screener, load_config, DEFAULT_CONFIG_PATH
        from utils.polygon_client import PolygonClient
        from utils.finnhub_client import FinnhubClient
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"import failed: {exc}")

    cfg = load_config(DEFAULT_CONFIG_PATH)
    polygon_key = cfg.get("polygon_api_key") or os.environ.get("POLYGON_API_KEY")
    if not polygon_key:
        raise HTTPException(status_code=400, detail="POLYGON_API_KEY not configured")
    try:
        polygon = PolygonClient(api_key=polygon_key)
        finnhub_key = cfg.get("finnhub_api_key") or os.environ.get("FINNHUB_API_KEY")
        finnhub = FinnhubClient(api_key=finnhub_key) if finnhub_key else None
        screener = Screener(cfg, polygon=polygon, finnhub=finnhub, store=store)
        hits = screener.scan()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"scan failed: {exc}")
    return {"hits": len(hits)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount static assets (CSS/JS) last so API routes take precedence.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
