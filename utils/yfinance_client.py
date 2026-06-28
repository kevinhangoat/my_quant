import datetime
import itertools
from typing import Optional, Sequence
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
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
        *,
        warmup_bars: int = 0,
    ):
        """Return price data for a ticker between two datetimes."""
        if start >= end:
            raise ValueError("start must be earlier than end")
        fetch_start = start
        if warmup_bars > 0:
            fetch_start = self._apply_warmup_start(start, interval, warmup_bars)
        return self.get_history(ticker, start=fetch_start, end=end, interval=interval)

    @staticmethod
    def _apply_warmup_start(start: datetime.datetime, interval: str, warmup_bars: int) -> datetime.datetime:
        delta = YFinanceClient._interval_to_timedelta(interval)
        return (pd.Timestamp(start) - warmup_bars * delta).to_pydatetime()

    @staticmethod
    def _interval_to_timedelta(interval: str) -> pd.Timedelta:
        import re
        match = re.match(r"^(\d+)([a-zA-Z]+)$", interval)
        if not match:
            raise ValueError(f"Unsupported interval: {interval}")
        value = int(match.group(1))
        unit = match.group(2).lower()
        if unit == "m":
            return pd.Timedelta(minutes=value)
        if unit == "h":
            return pd.Timedelta(hours=value)
        if unit == "d":
            return pd.Timedelta(days=value)
        if unit in ("w", "wk"):
            return pd.Timedelta(weeks=value)
        if unit in ("mo",):
            return pd.Timedelta(days=30 * value)
        raise ValueError(f"Unsupported interval: {interval}")

    # -------------------- plot_candlestick helpers --------------------

    DEFAULT_MA_WINDOWS = (5, 10, 20, 30, 48, 60, 120, 240)
    DEFAULT_MA_PALETTE = (
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
        "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    )
    DEFAULT_MACD_COLORS = {
        "diff": "#1F77B4",
        "dea": "#FF7F0E",
        "hist_up": "#26A69A",
        "hist_down": "#EF5350",
        "zero": "#9E9E9E",
    }

    @staticmethod
    def _align_tz(ts, tz):
        """Localize/convert a Timestamp to match `tz` (which may be None)."""
        if ts is None or tz is None:
            return ts
        ts = pd.Timestamp(ts)
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    @classmethod
    def _slice_display_window(cls, data, display_start, display_end):
        if display_start is None and display_end is None:
            return data
        tz = getattr(data.index, "tz", None)
        start_ts = cls._align_tz(display_start, tz) if display_start is not None else None
        end_ts = cls._align_tz(display_end, tz) if display_end is not None else None
        return data.loc[start_ts:end_ts]

    @staticmethod
    def _masked_series(index, mask, value):
        s = pd.Series(index=index, dtype="float64")
        s.loc[mask] = value
        return s

    @classmethod
    def _ma_series(cls, full_data, price_col, window, ma_type="sma"):
        """Return a moving-average series for `window`."""
        src = full_data[price_col]
        if str(ma_type).lower() == "ema":
            return src.ewm(span=window, adjust=False, min_periods=1).mean()
        return src.rolling(window=window, min_periods=1).mean()

    @classmethod
    def _build_ma_addplots(
        cls, full_data, plot_data, price_col, windows, palette, colors_map, ma_type="sma",
    ):
        windows = windows if windows is not None else cls.DEFAULT_MA_WINDOWS
        if not windows:
            return [], []
        palette = palette if palette is not None else cls.DEFAULT_MA_PALETTE
        color_cycle = itertools.cycle(palette)
        prefix = "EMA" if str(ma_type).lower() == "ema" else "MA"
        apds, labels = [], []
        for window in windows:
            if window <= 0:
                continue
            series = cls._ma_series(full_data, price_col, window, ma_type).loc[plot_data.index]
            if colors_map and window in colors_map:
                color = colors_map[window]
            else:
                color = next(color_cycle)
            apds.append(mpf.make_addplot(series, panel=0, color=color, width=1.0, label=f"{prefix}{window}"))
            labels.append((window, series, color))
        return apds, labels

    @classmethod
    def _build_ma_band_addplots(
        cls,
        full_data,
        plot_data,
        price_col,
        fill_between_mas,
        ma_type="sma",
        fast_above_color="#01F001",
        fast_below_color="#F80000",
        fill_alpha=0.18,
    ):
        """Fill the region between two MAs, coloured by which line is on top.

        `fill_between_mas` is a 2-tuple/list ``(fast_window, slow_window)``.
        Where fast>=slow the band is filled with ``fast_above_color``, otherwise
        with ``fast_below_color``.
        """
        if not fill_between_mas or len(fill_between_mas) != 2:
            return []
        fast_w, slow_w = fill_between_mas
        if fast_w <= 0 or slow_w <= 0:
            return []

        fast = cls._ma_series(full_data, price_col, fast_w, ma_type).loc[plot_data.index]
        slow = cls._ma_series(full_data, price_col, slow_w, ma_type).loc[plot_data.index]
        upper = fast.combine(slow, max)
        lower = fast.combine(slow, min)

        above_mask = (fast >= slow).fillna(False)
        below_mask = ~above_mask & fast.notna() & slow.notna()

        def _masked(series, mask):
            out = pd.Series(index=plot_data.index, dtype="float64")
            out.loc[mask] = series.loc[mask]
            return out

        apds = []
        if above_mask.any():
            up_hi = _masked(upper, above_mask)
            up_lo = _masked(lower, above_mask)
            apds.append(mpf.make_addplot(
                up_hi, panel=0, color=fast_above_color, alpha=0,
                fill_between=dict(y1=up_hi.values, y2=up_lo.values,
                                  alpha=fill_alpha, color=fast_above_color),
            ))
        if below_mask.any():
            dn_hi = _masked(upper, below_mask)
            dn_lo = _masked(lower, below_mask)
            apds.append(mpf.make_addplot(
                dn_hi, panel=0, color=fast_below_color, alpha=0,
                fill_between=dict(y1=dn_hi.values, y2=dn_lo.values,
                                  alpha=fill_alpha, color=fast_below_color),
            ))
        return apds

    @classmethod
    def _build_macd_addplots(cls, full_data, plot_data, price_col, fast, slow, signal, colors_override):
        close = full_data[price_col]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        diff_full = ema_fast - ema_slow
        dea_full = diff_full.ewm(span=signal, adjust=False).mean()
        hist_full = diff_full - dea_full
        diff = diff_full.loc[plot_data.index]
        dea = dea_full.loc[plot_data.index]
        hist = hist_full.loc[plot_data.index]

        colors = dict(cls.DEFAULT_MACD_COLORS)
        if colors_override:
            colors.update(colors_override)

        hist_colors = [colors["hist_up"] if v >= 0 else colors["hist_down"] for v in hist]
        zero_line = pd.Series(0.0, index=plot_data.index)
        apds = [
            mpf.make_addplot(hist, panel=1, type="bar", color=hist_colors, alpha=0.6),
            mpf.make_addplot(diff, panel=1, color=colors["diff"], width=1.0),
            mpf.make_addplot(dea, panel=1, color=colors["dea"], width=1.0),
            mpf.make_addplot(zero_line, panel=1, color=colors["zero"], width=0.6),
        ]
        labels = [("DIFF", diff, colors["diff"]), ("DEA", dea, colors["dea"])]
        return apds, labels

    @classmethod
    def _build_zone_addplots(cls, plot_data, zones, color):
        apds = []
        tz = getattr(plot_data.index, "tz", None)
        view_start, view_end = plot_data.index[0], plot_data.index[-1]
        for zone in zones:
            start_time = cls._align_tz(zone.end, tz)
            broken_time = cls._align_tz(zone.broken_time, tz) if zone.broken_time is not None else None
            low, high, strength = zone.lower, zone.upper, zone.strength

            if start_time < view_start:
                continue
            zone_start = max(start_time, view_start)
            zone_end = min(broken_time, view_end) if broken_time is not None else view_end
            mask = (plot_data.index >= zone_start) & (plot_data.index <= zone_end)
            if not mask.any():
                continue

            y_low = cls._masked_series(plot_data.index, mask, low)
            y_high = cls._masked_series(plot_data.index, mask, high)
            apds.append(mpf.make_addplot(
                y_high, panel=0, color=color, alpha=0,
                fill_between=dict(y1=y_high.values, y2=y_low.values, alpha=0.2, color=color),
            ))
            if strength is not None:
                zone_indices = plot_data.index[mask]
                mid_idx = zone_indices[len(zone_indices) // 2]
                strength_series = pd.Series(index=plot_data.index, dtype="float64")
                strength_series.loc[mid_idx] = low + (high - low) / 2
                apds.append(mpf.make_addplot(
                    strength_series, panel=0, type="scatter",
                    marker=f"${float(strength):.2f}$", markersize=260, color=color,
                ))
            apds.append(mpf.make_addplot(y_low, panel=0, color=color, linestyle="--", width=0.6))
            apds.append(mpf.make_addplot(y_high, panel=0, color=color, linestyle="--", width=0.6))
        return apds

    @classmethod
    def _filter_trades_to_window(cls, trades_df, display_start, display_end):
        if trades_df is None or trades_df.empty:
            return trades_df
        if display_start is None and display_end is None:
            return trades_df
        view = trades_df.copy()
        entry_times = pd.to_datetime(view["entry_time"])
        entry_tz = getattr(entry_times.dt, "tz", None)
        if display_start is not None:
            view = view[entry_times >= cls._align_tz(display_start, entry_tz)]
            entry_times = pd.to_datetime(view["entry_time"])
        if display_end is not None:
            view = view[entry_times <= cls._align_tz(display_end, entry_tz)]
        return view

    @classmethod
    def _build_trade_addplots(cls, plot_data, trades_view):
        apds = []
        if trades_view is None or trades_view.empty:
            return apds
        tz = getattr(plot_data.index, "tz", None)
        index = plot_data.index

        def _nearest_idx(ts):
            ts = cls._align_tz(ts, tz)
            pos = index.get_indexer([ts], method="nearest")
            return index[pos[0]] if pos[0] != -1 else None

        for _, trade in trades_view.iterrows():
            entry_idx = _nearest_idx(trade["entry_time"])
            if entry_idx is None:
                continue

            entry_price = trade["entry_price"]
            raw_side = str(trade.get("side", trade.get("direction", "buy"))).strip().lower()
            label = "BUY" if raw_side.startswith("l") else "SELL"
            color = "green" if label == "BUY" else "red"

            marker_series = pd.Series(index=index, dtype="float64")
            marker_series.loc[entry_idx] = entry_price
            apds.append(mpf.make_addplot(
                marker_series, panel=0, type="scatter",
                marker=f"${label}$", markersize=300, color=color,
            ))

            exit_time = trade.get("exit_time") or trade.get("close_time") or trade.get("end_time")
            end_idx = _nearest_idx(exit_time) if exit_time is not None else index[-1]
            if end_idx is None or end_idx < entry_idx:
                continue
            mask = (index >= entry_idx) & (index <= end_idx)
            if not mask.any():
                continue

            def _add_band(upper_val, lower_val, fill_color, alpha=0.18):
                if lower_val is None or upper_val is None:
                    return
                lo, hi = sorted([lower_val, upper_val])
                lo_s = cls._masked_series(index, mask, lo)
                hi_s = cls._masked_series(index, mask, hi)
                apds.append(mpf.make_addplot(
                    hi_s, panel=0, color=fill_color, alpha=0,
                    fill_between=dict(y1=hi_s.values, y2=lo_s.values, alpha=alpha, color=fill_color),
                ))

            _add_band(entry_price, trade.get("stop_loss"), "#020202")
            _add_band(entry_price, trade.get("take_profit"), "#07A3F8")
        return apds

    @staticmethod
    def _bind_addplots_to_axes(apds, ax, macd_ax):
        for apd in apds:
            if not isinstance(apd, dict) or apd.get("ax") is not None:
                continue
            panel = apd.get("panel", 0)
            apd["ax"] = macd_ax if (panel == 1 and macd_ax is not None) else ax

    @staticmethod
    def _annotate_series_ends(target_ax, plot_data, label_info):
        if target_ax is None or not label_info:
            return
        for label_text, series, color in label_info:
            if series.empty:
                continue
            last_val = series.iloc[-1]
            if pd.isna(last_val):
                continue
            target_ax.annotate(
                f" {label_text}",
                xy=(len(plot_data) - 1, last_val),
                xytext=(4, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
                ha="left",
                annotation_clip=False,
            )

    # ------------------------------------------------------------------

    def plot_candlestick(
        self,
        data,
        zones=[],
        trades_df=None,
        *,
        ax=None,
        macd_ax=None,
        show_mas: bool = True,
        ma_windows: Optional[Sequence[int]] = None,
        ma_colors: Optional[dict[int, str]] = None,
        ma_palette: Optional[Sequence[str]] = None,
        ma_price_col: str = "Close",
        ma_type: str = "sma",
        fill_between_mas: Optional[Sequence[int]] = None,
        fill_above_color: str = "#2CA02C",
        fill_below_color: str = "#D62728",
        fill_alpha: float = 0.18,
        show_macd: bool = True,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_price_col: str = "Close",
        macd_colors: Optional[dict[str, str]] = None,
        display_start: Optional[datetime.datetime] = None,
        display_end: Optional[datetime.datetime] = None,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """Plot a candlestick chart with optional supply/demand zones, trades, MAs and MACD."""

        if ax is not None and macd_ax is None and show_macd:
            show_macd = False

        plot_data = self._slice_display_window(data, display_start, display_end)
        apds = []
        ma_labels = []
        macd_labels = []

        # Trades
        trades_view = self._filter_trades_to_window(trades_df, display_start, display_end)
        apds += self._build_trade_addplots(plot_data, trades_view)

        # Zones
        if zones:
            supply = [z for z in zones if z.zone_type == "supply"]
            demand = [z for z in zones if z.zone_type == "demand"]
            apds += self._build_zone_addplots(plot_data, supply, "tab:red")
            apds += self._build_zone_addplots(plot_data, demand, "tab:green")

        # Indicators
        if show_mas:
            ma_apds, ma_labels = self._build_ma_addplots(
                data, plot_data, ma_price_col, ma_windows, ma_palette, ma_colors,
                ma_type=ma_type,
            )
            apds += ma_apds
        if fill_between_mas:
            apds += self._build_ma_band_addplots(
                data, plot_data, ma_price_col, fill_between_mas,
                ma_type=ma_type,
                fast_above_color=fill_above_color,
                fast_below_color=fill_below_color,
                fill_alpha=fill_alpha,
            )
        if show_macd:
            macd_apds, macd_labels = self._build_macd_addplots(
                data, plot_data, macd_price_col, macd_fast, macd_slow, macd_signal, macd_colors,
            )
            apds += macd_apds

        # Render
        plot_kwargs = dict(type="candle", addplot=apds)
        if ax is not None:
            self._bind_addplots_to_axes(apds, ax, macd_ax)
            plot_kwargs["ax"] = ax
        else:
            if show_macd:
                plot_kwargs["panel_ratios"] = (3, 1)
            if title:
                plot_kwargs["title"] = title
        mpf.plot(plot_data, **plot_kwargs)

        # End-of-series labels (only when caller supplied axes)
        if show_mas and ax is not None:
            prefix = "EMA" if str(ma_type).lower() == "ema" else "MA"
            self._annotate_series_ends(
                ax, plot_data, [(f"{prefix}{w}", s, c) for w, s, c in ma_labels],
            )
        if show_macd:
            self._annotate_series_ends(macd_ax, plot_data, macd_labels)

        if ax is not None and title:
            ax.set_title(title)
        if show:
            mpf.show()


    def plot_candlestick_grid(
        self,
        interval_views,
        *,
        display_start: Optional[datetime.datetime] = None,
        display_end: Optional[datetime.datetime] = None,
        show_mas: bool = True,
        show_macd: bool = True,
    ):
        """Plot up to four interval views in a 2x2 grid (with optional MACD panels)."""

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        rows_per_cell = 4 if show_macd else 3
        spacer_rows = 1
        total_rows = 2 * rows_per_cell + spacer_rows
        gs = GridSpec(total_rows, 2, figure=fig, hspace=0.3, wspace=0.05)

        cell_axes = []
        for cell in range(4):
            row_block = cell // 2
            col = cell % 2
            row_start = row_block * rows_per_cell + (spacer_rows if row_block > 0 else 0)
            if show_macd:
                price_ax = fig.add_subplot(gs[row_start:row_start + 3, col])
                macd_ax = fig.add_subplot(gs[row_start + 3, col], sharex=price_ax)
            else:
                price_ax = fig.add_subplot(gs[row_start:row_start + 3, col])
                macd_ax = None
            cell_axes.append((price_ax, macd_ax))

        for idx, view in enumerate(interval_views[:4]):
            price_ax, macd_ax = cell_axes[idx]
            self.plot_candlestick(
                view["data"],
                zones=view.get("zones", []),
                trades_df=view.get("trades_df"),
                ax=price_ax,
                macd_ax=macd_ax,
                show_mas=show_mas,
                show_macd=show_macd and macd_ax is not None,
                display_start=view.get("display_start", display_start),
                display_end=view.get("display_end", display_end),
                title=view.get("title"),
                show=False,
            )

        for idx in range(len(interval_views), 4):
            price_ax, macd_ax = cell_axes[idx]
            price_ax.axis("off")
            if macd_ax is not None:
                macd_ax.axis("off")

        plt.show()

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
