# polygon_client.py
"""
Thin Polygon.io REST client.
Falls back silently to yfinance if POLYGON_API_KEY is not set or a request fails.
Free Basic tier = 15-min delayed data.
Starter tier ($29/mo) = real-time data.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("POLYGON_API_KEY", "")
_BASE = "https://api.polygon.io"
_SESSION = requests.Session()
_TIMEOUT = 10

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 300  # 5 min


def _cached(key: str, ttl: int = _CACHE_TTL):
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry and time.time() - entry[0] < ttl:
            return entry[1]
    return None


def _store(key: str, value: object):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)
    return value


def available() -> bool:
    """True if a Polygon API key is configured."""
    return bool(_API_KEY)


def _get(path: str, params: dict | None = None) -> dict | None:
    if not _API_KEY:
        return None
    try:
        p = {"apiKey": _API_KEY, **(params or {})}
        r = _SESSION.get(f"{_BASE}{path}", params=p, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Snapshot — current price + basic stats for one ticker
# ---------------------------------------------------------------------------

def get_snapshot(ticker: str) -> dict | None:
    """
    Return a dict with current price, change %, volume, vwap, etc.
    Uses /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}
    """
    key = f"snap_{ticker}"
    cached = _cached(key, ttl=60)
    if cached is not None:
        return cached

    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    if not data or "ticker" not in data:
        return None

    t = data["ticker"]
    day = t.get("day", {})
    minute = t.get("min", {})   # most recent 1-min bar — most reliable current price
    prev = t.get("prevDay", {})
    # day.c is 0 pre-market; fall through to minute bar then lastTrade
    current_price = (
        (day.get("c") or None)
        or (minute.get("c") or None)
        or (t.get("lastTrade", {}).get("p") or None)
    )
    result = {
        "ticker": ticker,
        "price": current_price,
        "open": day.get("o") or minute.get("o"),
        "high": day.get("h") or minute.get("h"),
        "low": day.get("l") or minute.get("l"),
        "close": current_price,
        "volume": day.get("v") or minute.get("v"),
        "vwap": day.get("vw") or minute.get("vw"),
        "prev_close": prev.get("c"),
        "change_pct": t.get("todaysChangePerc"),
        "source": "polygon",
    }
    return _store(key, result)


def get_snapshots_batch(tickers: list[str]) -> dict[str, dict]:
    """
    Bulk snapshot for multiple tickers in one API call.
    Returns dict keyed by ticker symbol.
    """
    if not _API_KEY or not tickers:
        return {}

    key = f"snap_batch_{'_'.join(sorted(tickers[:10]))}"
    cached = _cached(key, ttl=60)
    if cached is not None:
        return cached

    data = _get(
        "/v2/snapshot/locale/us/markets/stocks/tickers",
        params={"tickers": ",".join(tickers)},
    )
    if not data or "tickers" not in data:
        return {}

    result = {}
    for t in data["tickers"]:
        sym = t.get("ticker", "")
        day = t.get("day", {})
        minute = t.get("min", {})
        prev = t.get("prevDay", {})
        current_price = (
            (day.get("c") or None)
            or (minute.get("c") or None)
            or (t.get("lastTrade", {}).get("p") or None)
        )
        result[sym] = {
            "ticker": sym,
            "price": current_price,
            "open": day.get("o") or minute.get("o"),
            "high": day.get("h") or minute.get("h"),
            "low": day.get("l") or minute.get("l"),
            "close": current_price,
            "volume": day.get("v"),
            "vwap": day.get("vw"),
            "prev_close": prev.get("c"),
            "change_pct": t.get("todaysChangePerc"),
            "source": "polygon",
        }
    return _store(key, result)


# ---------------------------------------------------------------------------
# Daily OHLCV bars — replaces yfinance daily history
# ---------------------------------------------------------------------------

def get_daily_bars(ticker: str, days: int = 365) -> pd.DataFrame | None:
    """
    Return a daily OHLCV DataFrame compatible with yfinance output shape.
    Columns: Open, High, Low, Close, Volume
    Index: DatetimeIndex (UTC)
    """
    key = f"daily_{ticker}_{days}"
    cached = _cached(key, ttl=3600)
    if cached is not None:
        return cached

    end = date.today()
    start = end - timedelta(days=days + 5)
    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        params={"adjusted": "true", "sort": "asc", "limit": 500},
    )
    if not data or "results" not in data or not data["results"]:
        return None

    rows = data["results"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("date").rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].tail(days)
    return _store(key, df)


# ---------------------------------------------------------------------------
# Intraday 5-min bars — replaces yfinance 5m history
# ---------------------------------------------------------------------------

def get_intraday_bars(ticker: str, days: int = 60) -> pd.DataFrame | None:
    """
    Return 5-min OHLCV bars for the last `days` calendar days.
    Index localised to America/New_York.
    """
    key = f"intra_{ticker}_{days}"
    cached = _cached(key, ttl=300)
    if cached is not None:
        return cached

    end = date.today()
    start = end - timedelta(days=days + 3)
    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/5/minute/{start}/{end}",
        params={"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    if not data or "results" not in data or not data["results"]:
        return None

    df = pd.DataFrame(data["results"])
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("datetime").rename(columns={
        "o": "Open", "h": "High", "l": "Low",
        "c": "Close", "v": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.tz_convert("America/New_York")
    return _store(key, df)


# ---------------------------------------------------------------------------
# Options snapshot — top-of-book for a ticker's options chain
# ---------------------------------------------------------------------------

def get_options_snapshot(ticker: str, expiration: str | None = None,
                          contract_type: str | None = None,
                          limit: int = 50) -> list[dict]:
    """
    Return options chain snapshot records from Polygon.
    expiration: 'YYYY-MM-DD' filter
    contract_type: 'call' or 'put'
    """
    key = f"opts_{ticker}_{expiration}_{contract_type}_{limit}"
    cached = _cached(key, ttl=60)
    if cached is not None:
        return cached

    params: dict = {"limit": limit, "sort": "expiration_date"}
    if expiration:
        params["expiration_date"] = expiration
    if contract_type:
        params["contract_type"] = contract_type

    data = _get(f"/v3/snapshot/options/{ticker}", params=params)
    if not data or "results" not in data:
        return []

    results = []
    for r in data["results"]:
        d = r.get("details", {})
        greeks = r.get("greeks", {})
        day = r.get("day", {})
        quote = r.get("last_quote", {})
        results.append({
            "ticker": d.get("ticker"),
            "underlying": ticker,
            "contract_type": d.get("contract_type"),
            "expiration": d.get("expiration_date"),
            "strike": d.get("strike_price"),
            "bid": quote.get("bid", 0),
            "ask": quote.get("ask", 0),
            "last_price": day.get("close", 0),
            "volume": day.get("volume", 0),
            "open_interest": r.get("open_interest", 0),
            "implied_volatility": r.get("implied_volatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "source": "polygon",
        })
    return _store(key, results)
