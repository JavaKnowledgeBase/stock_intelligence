"""
Congress Trade Tracker — fetches recent congressional stock disclosures.

Primary source: Quiver Quantitative free API (https://api.quiverquant.com).
Set QUIVER_API_KEY in Streamlit Cloud secrets (or .env locally).

Without an API key the module returns empty DataFrames and the dashboard
shows a helpful setup message instead of an error.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests

_BASE = "https://api.quiverquant.com/beta"
_TIMEOUT = 15


def _key() -> str | None:
    return os.getenv("QUIVER_API_KEY", "").strip() or None


def _headers() -> dict:
    return {"Accept": "application/json", "Authorization": f"Token {_key()}"}


def _get(path: str) -> list[dict]:
    try:
        resp = requests.get(f"{_BASE}{path}", headers=_headers(), timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json() or []
    except Exception:
        return []


def get_congress_trades(
    days_back: int = 60,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Fetch congressional stock transaction disclosures.

    Returns DataFrame with columns:
      trade_date, filed_date, politician, party, chamber,
      ticker, company, transaction, amount
    """
    if not _key():
        return pd.DataFrame()

    path = f"/live/congresstrading/{ticker}" if ticker else "/live/congresstrading"
    data = _get(path)
    if not data:
        return pd.DataFrame()

    cutoff = datetime.now() - timedelta(days=days_back)
    rows = []

    for item in data:
        try:
            # Quiver Quant field names (may vary slightly by endpoint version)
            trade_date_raw = (
                item.get("Date")
                or item.get("TradeDate")
                or item.get("Transaction_Date")
                or ""
            )
            filed_date_raw = (
                item.get("ReportDate")
                or item.get("Filed_Date")
                or ""
            )

            trade_date = pd.to_datetime(trade_date_raw, errors="coerce")
            if pd.notna(trade_date) and trade_date.to_pydatetime() < cutoff:
                continue

            politician = (
                item.get("Representative")
                or item.get("Senator")
                or item.get("Name")
                or "Unknown"
            )
            party_raw = item.get("Party") or ""
            party_symbol = (
                "🔴 R" if "R" in party_raw.upper()
                else "🔵 D" if "D" in party_raw.upper()
                else party_raw
            )

            amount_raw = item.get("Amount") or ""
            amount_str = _parse_amount(amount_raw)

            rows.append({
                "trade_date": trade_date_raw,
                "filed_date": filed_date_raw,
                "politician": politician,
                "party": party_symbol,
                "chamber": item.get("Chamber") or "",
                "ticker": str(item.get("Ticker") or "").upper(),
                "company": item.get("Company") or "",
                "transaction": _normalise_txn(item.get("Transaction") or ""),
                "amount": amount_str,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df


def _normalise_txn(txn: str) -> str:
    t = txn.strip().lower()
    if "purchase" in t or "buy" in t:
        return "🟢 Buy"
    if "sale" in t or "sell" in t:
        return "🔴 Sell"
    if "exchange" in t:
        return "🔄 Exchange"
    return txn.strip() or "Other"


def _parse_amount(raw: str) -> str:
    """Convert Quiver's amount codes to human-readable ranges."""
    mapping = {
        "1-15000": "$1K – $15K",
        "15001-50000": "$15K – $50K",
        "50001-100000": "$50K – $100K",
        "100001-250000": "$100K – $250K",
        "250001-500000": "$250K – $500K",
        "500001-1000000": "$500K – $1M",
        "1000001-5000000": "$1M – $5M",
        "5000001-25000000": "$5M – $25M",
        "25000001-50000000": "$25M – $50M",
    }
    return mapping.get(str(raw).strip(), str(raw))


def get_top_congress_tickers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Which stocks Congress is buying/selling most."""
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    ticker_df = df[df["ticker"].str.strip() != ""].copy()
    if ticker_df.empty:
        return pd.DataFrame()

    agg = (
        ticker_df.groupby("ticker")
        .agg(
            trades=("politician", "count"),
            buys=("transaction", lambda x: (x == "🟢 Buy").sum()),
            sells=("transaction", lambda x: (x == "🔴 Sell").sum()),
            politicians=("politician", lambda x: ", ".join(x.unique()[:3])),
        )
        .reset_index()
    )
    agg["sentiment"] = agg.apply(
        lambda r: "🟢 Bullish" if r["buys"] > r["sells"]
        else "🔴 Bearish" if r["sells"] > r["buys"]
        else "⚪ Mixed",
        axis=1,
    )
    return agg.sort_values("trades", ascending=False).head(top_n).reset_index(drop=True)


def get_most_active_politicians(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Most active traders in the dataset."""
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("politician")
        .agg(
            trades=("ticker", "count"),
            party=("party", "first"),
            chamber=("chamber", "first"),
            tickers=("ticker", lambda x: ", ".join(x.unique()[:5])),
            buys=("transaction", lambda x: (x == "🟢 Buy").sum()),
            sells=("transaction", lambda x: (x == "🔴 Sell").sum()),
        )
        .reset_index()
    )
    return agg.sort_values("trades", ascending=False).head(top_n).reset_index(drop=True)
