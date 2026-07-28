# my_quant

A personal quant toolkit for momentum day-trading and strategy backtesting.

It has two halves:

1. **Momentum screener + dashboard** — continuously scans the whole US stock
   market (via Polygon), fires alerts on Warrior-Trading-style momentum movers,
   scores them on 1-minute technicals, stores the results, and serves a local
   web dashboard.
2. **Backtesting** — replays trading strategies (supply/demand, EMA band, and
   the momentum plan) over historical data and reports simulated performance.

---

## Contents

| Path | What it is |
| --- | --- |
| [screener.py](screener.py) | Live momentum screener (3-stage funnel + alerts) |
| [momentum_backtest.py](momentum_backtest.py) | Backtest the momentum day-trading plan on Polygon history |
| [back_testing.py](back_testing.py) | Backtest the supply/demand and EMA-band strategies (yfinance) |
| [main.py](main.py) | Small example runner (Vegas tunnel on daily data) |
| `strategies/` | Strategy implementations (`moemteum.py`, `supply_and_demand.py`, `60_48.py`, ...) |
| `utils/` | Data clients (Polygon, Finnhub, yfinance), momentum scoring, storage, notifiers |
| `web_app/` | FastAPI dashboard + static frontend |
| `configs/` | JSON config files (screener, strategy params, tickers, test windows) |
| `data/` | Screener output, written per day (git-ignored) |

---

## Requirements

* **Python 3.7+** (3.9+ recommended — some modules use modern typing).
* A **Polygon.io** API key (a paid plan is needed for intraday minute bars and
  history used by the momentum score and backtest).
* Optional: a **Finnhub** API key for the news-catalyst and float criteria.

Install the screener / web / momentum dependencies:

```bash
pip install -r requirements-screener.txt
```

The backtester and plotting also need:

```bash
pip install yfinance mplfinance matplotlib
```

### API keys

Export them as environment variables (preferred — keeps secrets out of source):

```bash
export POLYGON_API_KEY=your_polygon_key
export FINNHUB_API_KEY=your_finnhub_key
```

They can also be set inside [configs/screener.json](configs/screener.json) under
`polygon_api_key` / `finnhub_api_key`.

---

## 1. Run the screener

The screener runs a 3-stage funnel so it scales to thousands of symbols:

* **Stage 1 — bulk pre-filter:** one Polygon full-market snapshot filters
  ~8,000 tickers using cheap criteria (price, % change, relative volume).
* **Stage 2 — enrich:** survivors get per-symbol Finnhub calls (news catalyst,
  float) plus a 1-minute **momentum score**.
* **Stage 3 — live watch:** a Polygon websocket streams survivors and fires
  alerts the instant a threshold is crossed.

```bash
# one funnel pass
python screener.py

# repeat the scan every scan_interval_sec
python screener.py --loop

# funnel once, then stream live websocket alerts
python screener.py --watch
```

Useful flags:

```bash
python screener.py --config configs/screener.json
python screener.py --polygon-key KEY --finnhub-key KEY
```

### Criteria (all configurable in [configs/screener.json](configs/screener.json))

| Criterion | Default |
| --- | --- |
| Relative volume | `>= 5x` |
| Day change | `>= 30%` |
| Price range | `$3 – $20` |
| Float | `<= 5,000,000` shares |
| News catalyst | keyword match within 24h (earnings, FDA, contract, acquisition, AI, ...) |

### Daily lists

Each confirmed hit is bucketed into:

* **Top gainers** — every hit passing the core criteria.
* **Small cap** — price below `small_cap_max_price` (default `$3`).
* **Low float** — float below `low_float_max_shares` (default `5M`).

### Momentum score (0–100)

For each hit, [utils/momentum.py](utils/momentum.py) computes EMA 9/20/48/60/200,
MACD, session-anchored VWAP and relative volume on the 1-minute chart, then a
weighted score. It rewards: very high relative volume, MACD above the zero line
(rising), the highest-volume bar being green, price above VWAP, and a bullish
EMA stack. Tune the weights via `momentum_weights` in the config.

### Alerts & storage

Alerts go to the notifiers enabled in the config (console by default; email,
Telegram, Pushover and webhook available). Results are cached as JSON under
`data/YYYY-MM-DD/` (`alerts.json`, `top_gainers.json`, `small_cap.json`,
`low_float.json`).

To enable **email** alerts, set the `email` notifier `enabled: true` in the
config and provide SMTP details, or use environment variables:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=you@gmail.com
export SMTP_PASSWORD=your_app_password
export EMAIL_FROM=you@gmail.com
export EMAIL_TO=you@gmail.com
```

---

## 2. Run the web dashboard

A FastAPI app serves the cached daily lists in an auto-refreshing,
Warrior-style table (score, symbol, price, change %, RVol, float, catalyst).

```bash
uvicorn web_app.app:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>.

* Pick a **day** and **list** (top gainers / small cap / low float).
* Toggle **auto-refresh** (polls every 15s).
* Click **Run scan** to trigger a fresh screener pass on demand (requires API
  keys configured).

---

## 3. Backtest the momentum plan

[momentum_backtest.py](momentum_backtest.py) replays the momentum day-trading
plan over Polygon history. Just like the live screener, it scans the **entire
US market every historical day** to find the symbols that would have triggered
that day, then simulates each trade:

1. For each day, pull the whole market's daily OHLCV in **one** grouped-daily
   call and apply the **same cheap screener criteria** (price range, day
   change, relative volume) to find the day's hits.
2. Only for those hits, fetch 1-minute bars and detect the intraday trigger.
3. Enter on a **pullback to EMA48/60**.
4. Scale out at **+10% / +30% / +60%**.
5. Stop out the remainder on a **−10%** drop from entry.

Relative volume comes from a rolling average daily volume seeded by a short
warmup of grouped-daily calls before the start date, then updated as each day is
processed — so no per-symbol history requests are needed.

```bash
export POLYGON_API_KEY=...

# scan the whole market across the date range
python momentum_backtest.py --start 2026-06-01 --end 2026-06-27
python momentum_backtest.py --start 2026-06-26 --end 2026-06-27 --plot

# optional: restrict the scanned universe for a faster/cheaper run
python momentum_backtest.py --start 2026-06-01 --end 2026-06-27 --symbols AAPL TSLA
```

It prints each day's screener hits and a per-trade table plus aggregate stats
(win rate, average/total return, symbols traded). With `--plot` it charts each
triggered session's 1-minute candles with the EMA stack, MACD and trade markers.
Plan parameters — including `rvol_avg_window`, `max_candidates_per_day`,
`min_price`/`max_price` and `include_otc` — live in
[configs/momentum_params.json](configs/momentum_params.json).

---

## 4. Backtest the other strategies

[back_testing.py](back_testing.py) backtests the supply/demand and EMA-band
strategies using yfinance data, with plotting and a simulated account.

```bash
# supply/demand strategy (default)
python back_testing.py usdchf --plot

# EMA 60/48 band strategy
python back_testing.py usdchf --strategy 60_48 --plot

# multiple tickers
python back_testing.py usdchf eurusd --strategy sad
```

Ticker definitions are in [configs/tickers.json](configs/tickers.json); the
date window and intervals are in the `configs/test_infos*.json` files.

---

## Configuration files

| File | Purpose |
| --- | --- |
| [configs/screener.json](configs/screener.json) | Screener criteria, momentum weights, notifiers, data dir |
| [configs/momentum_params.json](configs/momentum_params.json) | Momentum backtest entry/exit/stop parameters |
| [configs/tickers.json](configs/tickers.json) | Ticker symbol → data symbol + spread mapping |
| `configs/test_infos*.json` | Backtest date windows and intervals |

---

## Notes

* The `data/` directory is git-ignored; it is created automatically on the first
  screener run.
* The screener degrades gracefully without a Finnhub key (runs the cheap
  snapshot criteria only) and without notifier credentials (falls back to the
  console notifier).
* Extended-hours bars are included by default (`extended_hours: true`) so the
  screener and backtest cover pre/post-market momentum.
