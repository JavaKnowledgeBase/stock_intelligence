"""
Insider Trading Flow
Surfaces recent Form 4 insider buy/sell transactions.

Data sources (both free, no API key):
  - yfinance  : insider_transactions property (Yahoo Finance data)
  - SEC EDGAR : submissions API for Form 4 filing dates / links

Traders use this to find conviction buys by C-suite and directors,
which historically precede significant price moves.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

_HEADERS = {"User-Agent": "StockIntelligencePlatform/1.0 research@example.com"}

# ── CIK map (ticker → SEC CIK) ────────────────────────────────────────────────
_CIK_MAP: dict[str, str] = {}
_CIK_LOCK = threading.Lock()
_CIK_LOADED = False


def _ensure_cik_map() -> None:
    global _CIK_LOADED
    with _CIK_LOCK:
        if _CIK_LOADED:
            return
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        with _CIK_LOCK:
            for item in resp.json().values():
                ticker = item.get("ticker", "").upper()
                cik = str(item.get("cik_str", "")).zfill(10)
                if ticker:
                    _CIK_MAP[ticker] = cik
            _CIK_LOADED = True
    except Exception:
        pass


def get_cik(ticker: str) -> Optional[str]:
    _ensure_cik_map()
    return _CIK_MAP.get(ticker.upper())


def _edgar_link(ticker: str) -> str:
    cik = get_cik(ticker)
    if not cik:
        return ""
    return (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=10"
    )


# ── Per-ticker in-memory cache ────────────────────────────────────────────────
_TICKER_CACHE: dict[str, dict] = {}
_TICKER_CACHE_LOCK = threading.Lock()
_TICKER_CACHE_TTL = 3600  # 1 hour


def _cache_get(ticker: str) -> Optional[list[dict]]:
    with _TICKER_CACHE_LOCK:
        entry = _TICKER_CACHE.get(ticker)
        if entry and time.time() - entry["ts"] < _TICKER_CACHE_TTL:
            return entry["data"]
    return None


def _cache_set(ticker: str, data: list[dict]) -> list[dict]:
    with _TICKER_CACHE_LOCK:
        _TICKER_CACHE[ticker] = {"ts": time.time(), "data": data}
    return data


# ── Fetch insider transactions for one ticker ─────────────────────────────────
def _fetch_one(ticker: str, max_rows: int = 10) -> list[dict]:
    cached = _cache_get(ticker)
    if cached is not None:
        return cached

    rows: list[dict] = []
    try:
        t = yf.Ticker(ticker)
        df = t.insider_transactions

        if df is None or df.empty:
            return _cache_set(ticker, [])

        # Normalise columns — Yahoo Finance column names vary by version
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        col_map = {
            "shares": ["shares", "startposition"],
            "value": ["value"],
            "transaction": ["transaction", "text"],
            "insider": ["insider", "name"],
            "position": ["position", "relationship"],
            "date": ["startdate", "date"],
        }

        def _find_col(candidates: list[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        date_col = _find_col(col_map["date"])
        shares_col = _find_col(col_map["shares"])
        value_col = _find_col(col_map["value"])
        txn_col = _find_col(col_map["transaction"])
        insider_col = _find_col(col_map["insider"])
        position_col = _find_col(col_map["position"])

        edgar_url = _edgar_link(ticker)

        for _, row in df.head(max_rows).iterrows():
            txn_text = str(row.get(txn_col, "")).strip() if txn_col else ""
            # Classify as Buy / Sale / Other
            txn_lower = txn_text.lower()
            if any(w in txn_lower for w in ("purchase", "buy", "acquisition", "award")):
                txn_type = "Buy"
            elif any(w in txn_lower for w in ("sale", "sell", "dispose")):
                txn_type = "Sale"
            else:
                txn_type = "Other"

            shares = row.get(shares_col) if shares_col else None
            value = row.get(value_col) if value_col else None
            date_val = row.get(date_col) if date_col else None

            rows.append({
                "ticker": ticker,
                "date": str(pd.Timestamp(date_val).date()) if date_val else "",
                "insider": str(row.get(insider_col, "")).strip() if insider_col else "",
                "position": str(row.get(position_col, "")).strip() if position_col else "",
                "transaction": txn_type,
                "transaction_detail": txn_text,
                "shares": int(shares) if pd.notna(shares) and shares else None,
                "value_usd": int(value) if pd.notna(value) and value else None,
                "edgar_link": edgar_url,
            })

    except Exception:
        pass

    return _cache_set(ticker, rows)


def get_insider_flow(
    tickers: list[str],
    max_per_ticker: int = 5,
    transaction_filter: str = "All",
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Return a combined DataFrame of recent insider transactions.

    transaction_filter: "All" | "Buy" | "Sale"
    """
    _ensure_cik_map()
    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, t, max_per_ticker): t for t in tickers
        }
        for f in as_completed(futures, timeout=90):
            try:
                all_rows.extend(f.result())
            except Exception:
                pass

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date", ascending=False)

    if transaction_filter in ("Buy", "Sale"):
        df = df[df["transaction"] == transaction_filter]

    # Format value column for display
    if "value_usd" in df.columns:
        df["value_fmt"] = df["value_usd"].apply(
            lambda v: f"${v:,.0f}" if pd.notna(v) and v else "—"
        )
    if "shares" in df.columns:
        df["shares_fmt"] = df["shares"].apply(
            lambda s: f"{s:,}" if pd.notna(s) and s else "—"
        )

    return df.reset_index(drop=True)
