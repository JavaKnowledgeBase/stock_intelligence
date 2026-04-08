# timing_engine.py
"""
Intraday Timing Engine
Estimates optimal buy/sell windows for a ticker based on 30 days of
5-minute historical OHLCV data from yfinance (free, no API key needed).

Key outputs per ticker:
- best_entry_window  : recommended entry time slot (e.g. "10:00–10:30 AM ET")
- best_exit_window   : recommended exit / take-profit time slot
- gap_fade_prob      : probability a gap (>0.5%) reverses within 30 min
- volume_profile     : relative volume per 30-min slot (1.0 = daily avg)
- avg_return_by_slot : average 30-min return per slot over last 30 days
- timing_note        : plain-English recommendation
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import polygon_client as _polygon
except Exception:
    _polygon = None

# ---------------------------------------------------------------------------
# Time slots (ET, 30-min buckets covering regular session)
# ---------------------------------------------------------------------------

_SLOTS: list[tuple[str, str]] = [
    ("09:30", "10:00"),
    ("10:00", "10:30"),
    ("10:30", "11:00"),
    ("11:00", "11:30"),
    ("11:30", "12:00"),
    ("12:00", "12:30"),
    ("12:30", "13:00"),
    ("13:00", "13:30"),
    ("13:30", "14:00"),
    ("14:00", "14:30"),
    ("14:30", "15:00"),
    ("15:00", "15:30"),
    ("15:30", "16:00"),
]

_SLOT_LABELS = [f"{s}–{e} ET" for s, e in _SLOTS]

# Early session = first 3 slots (09:30–11:00); late = last 2 (15:00–16:00)
_EARLY_IDX = [0, 1, 2]
_LATE_IDX = [11, 12]
_MID_IDX = [3, 4, 5, 6, 7, 8, 9, 10]

# ---------------------------------------------------------------------------
# In-memory cache (keyed by ticker, expires after 4 hours)
# ---------------------------------------------------------------------------

_CACHE_TTL = 4 * 3600
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()


def _get_cached(ticker: str) -> dict | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(ticker)
        if entry and time.time() - entry[0] < _CACHE_TTL:
            return entry[1]
    return None


def _set_cached(ticker: str, result: dict) -> dict:
    with _CACHE_LOCK:
        _CACHE[ticker] = (time.time(), result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OHLCV = {"Open", "High", "Low", "Close", "Volume"}

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns and convert index to ET."""
    if isinstance(df.columns, pd.MultiIndex):
        # Find the level that contains OHLCV names
        for level in range(df.columns.nlevels):
            vals = set(df.columns.get_level_values(level))
            if _OHLCV & vals:
                df.columns = df.columns.get_level_values(level)
                break
        else:
            df.columns = df.columns.get_level_values(0)
    # Keep only OHLCV columns to avoid stray Series on iloc
    keep = [c for c in df.columns if c in _OHLCV]
    df = df[keep].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    return df


def _slot_index(time_str: str) -> int:
    """Return the _SLOTS index that contains this HH:MM time, or -1."""
    for i, (start, end) in enumerate(_SLOTS):
        if start <= time_str < end:
            return i
    return -1


def _slot_return(day_df: pd.DataFrame, slot_idx: int) -> float | None:
    """Compute open-to-close return for one slot on one day's data."""
    s_start, s_end = _SLOTS[slot_idx]
    bars = day_df.between_time(s_start, s_end, inclusive="left")
    if bars.empty or len(bars) < 1:
        return None
    open_px = float(bars["Open"].iloc[0])
    close_px = float(bars["Close"].iloc[-1])
    if open_px <= 0:
        return None
    return (close_px / open_px - 1) * 100


def _slot_volume(day_df: pd.DataFrame, slot_idx: int) -> float:
    """Total volume in one slot."""
    s_start, s_end = _SLOTS[slot_idx]
    bars = day_df.between_time(s_start, s_end, inclusive="left")
    return float(bars["Volume"].sum()) if not bars.empty else 0.0


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyse(df: pd.DataFrame) -> dict:
    """
    Given a 5-min OHLCV DataFrame (ET-localised, single ticker),
    compute slot-level statistics across all trading days.
    """
    n_slots = len(_SLOTS)

    slot_returns: list[list[float]] = [[] for _ in range(n_slots)]
    slot_volumes: list[list[float]] = [[] for _ in range(n_slots)]
    gap_samples: list[float] = []   # gap size (%)
    gap_fades: list[bool] = []      # did it fade within 30 min?

    trading_days = sorted(set(df.index.date))

    prev_close = None
    for day in trading_days:
        day_df = df[df.index.date == day].copy()
        if day_df.empty:
            continue

        # Gap analysis: first bar open vs prior day close
        first_open = float(day_df["Open"].iloc[0])
        if prev_close and prev_close > 0:
            gap_pct = (first_open / prev_close - 1) * 100
            if abs(gap_pct) >= 0.5:
                gap_samples.append(gap_pct)
                # Fade = price returned toward prior close within first 30 min
                slot0 = day_df.between_time("09:30", "10:00", inclusive="left")
                if not slot0.empty:
                    low_30 = float(slot0["Low"].min())
                    high_30 = float(slot0["High"].max())
                    faded = (
                        (gap_pct > 0 and low_30 <= prev_close * 1.001)
                        or (gap_pct < 0 and high_30 >= prev_close * 0.999)
                    )
                    gap_fades.append(faded)

        last_close_row = day_df["Close"].dropna()
        if not last_close_row.empty:
            prev_close = float(last_close_row.iloc[-1])

        # Per-slot stats
        for i in range(n_slots):
            ret = _slot_return(day_df, i)
            if ret is not None:
                slot_returns[i].append(ret)
            vol = _slot_volume(day_df, i)
            slot_volumes[i].append(vol)

    # Aggregate
    avg_returns = [
        round(float(np.mean(r)), 3) if r else 0.0
        for r in slot_returns
    ]
    win_rates = [
        round(sum(1 for r in lst if r > 0) / len(lst) * 100, 1) if lst else 50.0
        for lst in slot_returns
    ]
    avg_vols = [
        float(np.mean(v)) if v else 0.0
        for v in slot_volumes
    ]

    daily_avg_vol = sum(avg_vols) / len([v for v in avg_vols if v > 0]) if any(v > 0 for v in avg_vols) else 1.0
    vol_ratios = [
        round(v / daily_avg_vol, 2) if daily_avg_vol > 0 else 1.0
        for v in avg_vols
    ]

    gap_fade_prob = (
        round(sum(gap_fades) / len(gap_fades) * 100, 1)
        if gap_fades else None
    )

    return {
        "avg_returns": avg_returns,
        "win_rates": win_rates,
        "vol_ratios": vol_ratios,
        "gap_fade_prob": gap_fade_prob,
        "n_days": len(trading_days),
    }


def _pick_entry_window(avg_returns: list[float], vol_ratios: list[float],
                        contract_type: str) -> int:
    """
    Return the slot index that is the best entry for calls or puts,
    prioritising early-to-mid session slots with above-average volume.
    """
    candidates = _EARLY_IDX + _MID_IDX[:4]  # 09:30–13:00
    sign = 1 if contract_type == "call" else -1

    best_idx = candidates[0]
    best_score = -999.0
    for i in candidates:
        if vol_ratios[i] < 0.5:
            continue  # skip illiquid slots
        score = avg_returns[i] * sign + vol_ratios[i] * 0.3
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _pick_exit_window(avg_returns: list[float], vol_ratios: list[float],
                       entry_idx: int, contract_type: str) -> int:
    """
    Return the best take-profit / exit slot after the entry slot.
    For calls: look for the slot after entry where the cumulative move peaks.
    For puts: same but inverted.
    """
    sign = 1 if contract_type == "call" else -1
    look_ahead = list(range(entry_idx + 1, len(_SLOTS)))

    if not look_ahead:
        return len(_SLOTS) - 1

    # Cumulative return from entry
    cum_ret = 0.0
    peak_ret = -999.0
    peak_idx = look_ahead[0]
    for i in look_ahead:
        cum_ret += avg_returns[i] * sign
        if cum_ret > peak_ret:
            peak_ret = cum_ret
            peak_idx = i

    return peak_idx


def _build_note(ticker: str, contract_type: str, entry_idx: int, exit_idx: int,
                avg_returns: list[float], vol_ratios: list[float],
                win_rates: list[float], gap_fade_prob: float | None,
                n_days: int) -> str:
    """Compose a plain-English timing recommendation."""
    entry_label = _SLOT_LABELS[entry_idx]
    exit_label = _SLOT_LABELS[exit_idx]
    direction = "calls" if contract_type == "call" else "puts"
    entry_ret = avg_returns[entry_idx]
    entry_winrate = win_rates[entry_idx]
    entry_vol = vol_ratios[entry_idx]

    lines = [
        f"Based on {n_days} trading days of 5-min data:",
        f"• Best entry window for {direction}: **{entry_label}** "
        f"(avg slot return {entry_ret:+.2f}%, win rate {entry_winrate:.0f}%, "
        f"vol {entry_vol:.1f}× avg)",
        f"• Suggested exit / take-profit window: **{exit_label}**",
    ]

    # Open caution
    open_vol = vol_ratios[0]
    if open_vol >= 2.5:
        lines.append(
            f"• Opening slot (09:30–10:00) has very high volume ({open_vol:.1f}× avg) — "
            "spreads are wide and moves are erratic. Avoid entering in the first 5–10 minutes."
        )

    # Gap fade
    if gap_fade_prob is not None:
        if gap_fade_prob >= 60:
            lines.append(
                f"• Gap fade probability: {gap_fade_prob:.0f}% — "
                "gaps (≥0.5%) tend to reverse here. "
                f"If {ticker} gaps {'up' if contract_type == 'put' else 'down'} at open, "
                "wait for the fade before entering."
            )
        elif gap_fade_prob <= 35:
            lines.append(
                f"• Gap continuation: {100 - gap_fade_prob:.0f}% of gaps hold direction. "
                "An opening gap aligned with your trade bias can be traded with momentum."
            )

    # Lunch warning
    lunch_avg = float(np.mean([avg_returns[i] for i in [5, 6]]))
    if abs(lunch_avg) < 0.05:
        lines.append(
            "• Midday (12:00–13:00) is historically flat — avoid new entries during lunch."
        )

    # Late-day
    close_ret = avg_returns[-1]
    if abs(close_ret) >= 0.1:
        sign_str = "bullish" if close_ret > 0 else "bearish"
        lines.append(
            f"• Final 30 min (15:30–16:00) has a {sign_str} bias "
            f"({close_ret:+.2f}% avg) — {'calls' if close_ret > 0 else 'puts'} "
            "can benefit from late-day momentum."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_intraday_timing(ticker: str, contract_type: str = "call") -> dict:
    """
    Return intraday timing analysis for a single ticker.

    Parameters
    ----------
    ticker        : stock symbol
    contract_type : 'call' or 'put'

    Returns
    -------
    dict with keys:
        best_entry_window  str
        best_exit_window   str
        gap_fade_prob      float | None   (% probability)
        volume_profile     list[float]    (vol ratio per slot)
        avg_return_by_slot list[float]    (avg % return per slot)
        win_rate_by_slot   list[float]    (% of days positive per slot)
        slot_labels        list[str]
        timing_note        str
        n_days             int
        error              str | None
    """
    cached = _get_cached(f"{ticker}_{contract_type}")
    if cached is not None:
        return cached

    empty = {
        "best_entry_window": "10:00–10:30 ET",
        "best_exit_window": "11:00–11:30 ET",
        "gap_fade_prob": None,
        "volume_profile": [1.0] * len(_SLOTS),
        "avg_return_by_slot": [0.0] * len(_SLOTS),
        "win_rate_by_slot": [50.0] * len(_SLOTS),
        "slot_labels": _SLOT_LABELS,
        "timing_note": "Insufficient data to compute timing.",
        "n_days": 0,
        "error": None,
    }

    try:
        # Try Polygon first for intraday bars
        df = None
        if _polygon and _polygon.available():
            poly_df = _polygon.get_intraday_bars(ticker, days=60)
            if poly_df is not None and len(poly_df) >= 50:
                df = poly_df

        # Fallback to yfinance
        if df is None:
            raw = yf.download(
                ticker, period="60d", interval="5m",
                progress=False, auto_adjust=True, threads=False,
            )
            if raw is None or raw.empty or len(raw) < 50:
                empty["error"] = "Insufficient intraday history."
                return _set_cached(f"{ticker}_{contract_type}", empty)
            df = _flatten(raw)

        # Restrict to regular session 09:30–16:00
        df = df.between_time("09:30", "16:00")
        if df.empty:
            empty["error"] = "No regular-session bars found."
            return _set_cached(f"{ticker}_{contract_type}", empty)

        stats = _analyse(df)

        entry_idx = _pick_entry_window(
            stats["avg_returns"], stats["vol_ratios"], contract_type
        )
        exit_idx = _pick_exit_window(
            stats["avg_returns"], stats["vol_ratios"], entry_idx, contract_type
        )

        note = _build_note(
            ticker, contract_type, entry_idx, exit_idx,
            stats["avg_returns"], stats["vol_ratios"],
            stats["win_rates"], stats["gap_fade_prob"], stats["n_days"],
        )

        result = {
            "best_entry_window": _SLOT_LABELS[entry_idx],
            "best_exit_window": _SLOT_LABELS[exit_idx],
            "gap_fade_prob": stats["gap_fade_prob"],
            "volume_profile": stats["vol_ratios"],
            "avg_return_by_slot": stats["avg_returns"],
            "win_rate_by_slot": stats["win_rates"],
            "slot_labels": _SLOT_LABELS,
            "timing_note": note,
            "n_days": stats["n_days"],
            "error": None,
        }
        return _set_cached(f"{ticker}_{contract_type}", result)

    except Exception as exc:
        empty["error"] = str(exc)
        return empty


def get_intraday_timing_batch(
    ticker_contract_pairs: list[tuple[str, str]],
    workers: int = 6,
) -> dict[tuple[str, str], dict]:
    """
    Fetch timing for multiple (ticker, contract_type) pairs in parallel.
    Returns a dict keyed by (ticker, contract_type).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(get_intraday_timing, ticker, ct): (ticker, ct)
            for ticker, ct in ticker_contract_pairs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                ticker, ct = key
                results[key] = {
                    "best_entry_window": "10:00–10:30 ET",
                    "best_exit_window": "11:00–11:30 ET",
                    "gap_fade_prob": None,
                    "volume_profile": [1.0] * len(_SLOTS),
                    "avg_return_by_slot": [0.0] * len(_SLOTS),
                    "win_rate_by_slot": [50.0] * len(_SLOTS),
                    "slot_labels": _SLOT_LABELS,
                    "timing_note": f"Error: {exc}",
                    "n_days": 0,
                    "error": str(exc),
                }
    return results
