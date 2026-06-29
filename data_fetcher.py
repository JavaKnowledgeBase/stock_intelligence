"""
data_fetcher.py — Bulk price downloader with session cache.

The single biggest speed lever in the app: replacing 474 individual
yf.Ticker().history() calls (93s) with one yf.download() call (3-5s).

How it works:
  1. bulk_preload(tickers)  — called once before any scan starts.
     Downloads OHLCV for all tickers in batches of 200 via yf.download().
  2. get(ticker)            — returns the cached DataFrame instantly (0 ms).
     Falls back to individual fetch if ticker missed the bulk call.
  3. Cache TTL = 5 minutes. Auto-invalidated on next bulk_preload().

Thread-safe: all cache reads/writes use a RLock so 50 parallel workers
can call get() concurrently without corruption.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import pandas as pd
import yfinance as yf

_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_TS: float = 0.0
_CACHE_PERIOD: str = ""
_LOCK = threading.RLock()
_TTL = 300.0          # 5 minutes
_BATCH = 200          # tickers per yf.download() call — sweet spot for speed
_PRICE_COLS = {"Open", "High", "Low", "Close", "Volume"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise MultiIndex columns from yf.download(group_by='ticker')."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    for level in range(df.columns.nlevels):
        if _PRICE_COLS & set(df.columns.get_level_values(level)):
            df.columns = df.columns.get_level_values(level)
            return df
    df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def _is_fresh() -> bool:
    return bool(_CACHE) and (time.time() - _CACHE_TS) < _TTL


def _store(ticker: str, df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        with _LOCK:
            _CACHE[ticker.upper()] = df


# ── Public API ────────────────────────────────────────────────────────────────

def bulk_preload(
    tickers: list[str],
    period: str = "1y",
    force: bool = False,
) -> int:
    """
    Download OHLCV for *all* tickers in as few API calls as possible.

    Returns the number of tickers successfully loaded.
    Skips the download if the cache is still fresh (< 5 min old) and
    force=False — so calling this from multiple tabs is safe and free.
    """
    global _CACHE_TS, _CACHE_PERIOD

    with _LOCK:
        if not force and _is_fresh() and _CACHE_PERIOD == period:
            return len(_CACHE)

    unique = list(dict.fromkeys(t.upper() for t in tickers if t.strip()))
    loaded = 0

    for i in range(0, len(unique), _BATCH):
        batch = unique[i : i + _BATCH]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,    # yfinance uses its own internal thread pool here
            )
            if raw is None or raw.empty:
                continue

            if len(batch) == 1:
                flat = _flatten(raw.copy())
                flat = flat.dropna(how="all")
                if not flat.empty:
                    _store(batch[0], flat)
                    loaded += 1
            else:
                for ticker in batch:
                    try:
                        sub = raw[ticker].copy()
                        sub = _flatten(sub)
                        sub = sub.dropna(how="all")
                        if not sub.empty and len(sub) >= 5:
                            _store(ticker, sub)
                            loaded += 1
                    except (KeyError, TypeError):
                        pass
        except Exception:
            pass

    with _LOCK:
        _CACHE_TS = time.time()
        _CACHE_PERIOD = period

    return loaded


def get(
    ticker: str,
    period: str = "1y",
    min_rows: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Return OHLCV DataFrame for ticker.

    Checks the bulk cache first (0 ms). If not found, falls back to an
    individual yf.Ticker().history() call. Trims to the requested period
    when data was pre-loaded with a longer window.
    """
    key = ticker.upper()

    with _LOCK:
        df = _CACHE.get(key)

    if df is not None and not df.empty:
        # Trim to requested period if cache holds a longer window
        df = _trim_to_period(df, period)
        if len(df) >= min_rows:
            return df.copy()

    # Individual fallback
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval="1d", auto_adjust=True
        )
        if df is not None and not df.empty and len(df) >= min_rows:
            _store(key, df)
            return df.copy()
    except Exception:
        pass
    return None


def invalidate() -> None:
    """Force cache clear — next bulk_preload() will re-download everything."""
    global _CACHE_TS
    with _LOCK:
        _CACHE.clear()
        _CACHE_TS = 0.0


def cache_size() -> int:
    with _LOCK:
        return len(_CACHE)


def cache_age_seconds() -> float:
    return max(0.0, time.time() - _CACHE_TS) if _CACHE_TS else float("inf")


# ── Period trimming ───────────────────────────────────────────────────────────

_PERIOD_DAYS = {
    "1mo": 35, "3mo": 95, "6mo": 185,
    "1y": 370, "2y": 740, "5y": 1830,
}


def _trim_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = _PERIOD_DAYS.get(period)
    if days is None:
        return df
    cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=days)
    trimmed = df[df.index >= cutoff]
    return trimmed if not trimmed.empty else df
