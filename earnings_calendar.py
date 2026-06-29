"""
Earnings Calendar Engine
Fetches upcoming earnings events with IV / HV context for options traders.
Traders use this to find elevated-IV setups before announcements.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import yfinance as yf

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 3600  # 1 hour per ticker


def _cache_get(ticker: str) -> Optional[dict]:
    with _CACHE_LOCK:
        entry = _CACHE.get(ticker)
        if entry and time.time() - entry["ts"] < _CACHE_TTL:
            return entry["data"]
    return None


def _cache_set(ticker: str, data: Optional[dict]) -> Optional[dict]:
    with _CACHE_LOCK:
        _CACHE[ticker] = {"ts": time.time(), "data": data}
    return data


def _fetch_one(ticker: str) -> Optional[dict]:
    cached = _cache_get(ticker)
    if cached is not None:
        return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # ── Earnings date ──────────────────────────────────────────────────
        earnings_date: Optional[pd.Timestamp] = None
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                if isinstance(dates, list) and dates:
                    earnings_date = pd.Timestamp(dates[0])
                elif dates:
                    earnings_date = pd.Timestamp(dates)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.index:
                    earnings_date = pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
        except Exception:
            pass

        today = pd.Timestamp.now().normalize()
        if earnings_date is None or earnings_date.normalize() < today:
            return _cache_set(ticker, None)

        # ── EPS & Revenue estimates ────────────────────────────────────────
        eps_estimate = None
        rev_estimate = None
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                eps_estimate = cal.get("EPS Estimate") or cal.get("Earnings Average")
                rev_estimate = cal.get("Revenue Estimate") or cal.get("Revenue Average")
        except Exception:
            pass

        # ── Implied volatility (median of nearest expiry ATM calls) ───────
        iv_median: Optional[float] = None
        try:
            exps = t.options
            if exps:
                chain = t.option_chain(exps[0])
                calls_iv = chain.calls["impliedVolatility"].dropna()
                if not calls_iv.empty:
                    iv_median = round(float(calls_iv.median()) * 100, 1)
        except Exception:
            pass

        # ── Historical volatility (30-day annualised) ─────────────────────
        hv_30: Optional[float] = None
        try:
            hist = t.history(period="3mo", interval="1d")
            if len(hist) >= 20:
                rets = hist["Close"].pct_change().dropna()
                hv_30 = round(float(rets.std() * (252 ** 0.5) * 100), 1)
        except Exception:
            pass

        iv_hv_ratio = round(iv_median / hv_30, 2) if iv_median and hv_30 else None
        iv_elevated = (iv_hv_ratio or 0) >= 1.25

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        result = {
            "ticker": ticker,
            "earnings_date": str(earnings_date.date()),
            "days_until": int((earnings_date.normalize() - today).days),
            "eps_estimate": eps_estimate,
            "rev_estimate_b": round(rev_estimate / 1e9, 2) if rev_estimate else None,
            "iv_median_pct": iv_median,
            "hv_30_pct": hv_30,
            "iv_hv_ratio": iv_hv_ratio,
            "iv_elevated": iv_elevated,
            "price": round(float(price), 2) if price else None,
            "market_cap_b": round(market_cap / 1e9, 1) if market_cap else None,
            "sector": sector,
            "industry": industry,
        }
        return _cache_set(ticker, result)

    except Exception:
        return _cache_set(ticker, None)


def get_earnings_calendar(
    tickers: list[str],
    max_days_ahead: int = 30,
    max_workers: int = 12,
) -> pd.DataFrame:
    """
    Return a DataFrame of upcoming earnings sorted by days_until.
    Filters to events within max_days_ahead calendar days.
    """
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for f in as_completed(futures, timeout=90):
            try:
                result = f.result()
                if result:
                    rows.append(result)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["days_until"] >= 0]
    df = df[df["days_until"] <= max_days_ahead]
    df = df.sort_values("days_until").reset_index(drop=True)
    return df
