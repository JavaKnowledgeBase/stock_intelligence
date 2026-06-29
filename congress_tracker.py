"""
Congress Trade Tracker — fetches STOCK Act disclosures from two free public APIs:

  House: https://housestockwatcher.com/api  (JSON, no key)
  Senate: https://senatestockwatcher.com/api (JSON, no key)

Both sites aggregate the official government PTR filings daily.
No API key or registration needed.
"""

from datetime import datetime, timedelta

import pandas as pd
import requests

_TIMEOUT = 20

# S3 aggregate endpoints backed by the two community projects
_HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

# Fallback: direct site API endpoints
_HOUSE_API = "https://housestockwatcher.com/api"
_SENATE_API = "https://senatestockwatcher.com/api"


def _fetch_json(primary: str, fallback: str) -> list[dict]:
    for url in (primary, fallback):
        try:
            resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            # Both endpoints return either a list or {"data": [...]}
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", []) or data.get("transactions", []) or []
        except Exception:
            continue
    return []


def _normalise_txn(txn: str) -> str:
    t = (txn or "").strip().lower()
    if "purchase" in t or "buy" in t:
        return "🟢 Buy"
    if "sale" in t or "sell" in t:
        return "🔴 Sell"
    if "exchange" in t:
        return "🔄 Exchange"
    return (txn or "Other").strip()


def _parse_house(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for item in raw:
        try:
            rows.append({
                "trade_date": item.get("transaction_date", ""),
                "filed_date": item.get("disclosure_date", ""),
                "politician": item.get("representative", ""),
                "party": item.get("party", ""),
                "chamber": "House",
                "state": item.get("state", ""),
                "ticker": str(item.get("ticker") or "").upper().strip(),
                "company": item.get("asset_description", ""),
                "transaction": _normalise_txn(item.get("type", "")),
                "amount": item.get("amount", ""),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _parse_senate(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for item in raw:
        try:
            # Senate watcher nests under each senator object
            senator = item.get("Senator") or item.get("senator") or item.get("name", "")
            party = item.get("Party") or item.get("party", "")
            state = item.get("State") or item.get("state", "")

            # Transactions may be a list inside each senator object
            txns = item.get("transactions") or item.get("Transactions")
            if txns and isinstance(txns, list):
                for txn in txns:
                    rows.append({
                        "trade_date": txn.get("transaction_date", ""),
                        "filed_date": txn.get("disclosure_date", ""),
                        "politician": senator,
                        "party": party,
                        "chamber": "Senate",
                        "state": state,
                        "ticker": str(txn.get("ticker") or "").upper().strip(),
                        "company": txn.get("asset_description", ""),
                        "transaction": _normalise_txn(txn.get("type", "")),
                        "amount": txn.get("amount", ""),
                    })
            else:
                # Flat format
                rows.append({
                    "trade_date": item.get("transaction_date", ""),
                    "filed_date": item.get("disclosure_date", ""),
                    "politician": senator,
                    "party": party,
                    "chamber": "Senate",
                    "state": state,
                    "ticker": str(item.get("ticker") or "").upper().strip(),
                    "company": item.get("asset_description", ""),
                    "transaction": _normalise_txn(item.get("type", "")),
                    "amount": item.get("amount", ""),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def get_congress_trades(
    days_back: int = 60,
    ticker: str | None = None,
    chamber: str = "Both",
) -> pd.DataFrame:
    """
    Fetch recent congressional STOCK Act disclosures.

    Parameters
    ----------
    days_back : int
        How many calendar days of history to return.
    ticker : str | None
        Filter to a specific ticker (case-insensitive).
    chamber : str
        "Both", "House", or "Senate".

    Returns
    -------
    DataFrame with columns:
        trade_date, filed_date, politician, party, chamber, state,
        ticker, company, transaction, amount
    """
    frames = []

    if chamber in ("Both", "House"):
        raw_house = _fetch_json(_HOUSE_URL, _HOUSE_API)
        if raw_house:
            frames.append(_parse_house(raw_house))

    if chamber in ("Both", "Senate"):
        raw_senate = _fetch_json(_SENATE_URL, _SENATE_API)
        if raw_senate:
            frames.append(_parse_senate(raw_senate))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Clean up
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")

    # Date filter
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    df = df[df["trade_date"] >= cutoff]

    # Ticker filter
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]

    # Party emojis
    def _party_badge(p: str) -> str:
        p = (p or "").strip().upper()
        if p in ("R", "REPUBLICAN"):
            return "🔴 R"
        if p in ("D", "DEMOCRAT", "DEMOCRATIC"):
            return "🔵 D"
        if p in ("I", "INDEPENDENT"):
            return "⚪ I"
        return p or "?"

    df["party"] = df["party"].apply(_party_badge)

    df = df.sort_values("trade_date", ascending=False).reset_index(drop=True)
    return df


def get_top_congress_tickers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Which stocks Congress is buying/selling most."""
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    valid = df[df["ticker"].str.strip().ne("") & df["ticker"].ne("--")].copy()
    if valid.empty:
        return pd.DataFrame()

    agg = (
        valid.groupby("ticker")
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
            state=("state", "first"),
            tickers=("ticker", lambda x: ", ".join(x.unique()[:5])),
            buys=("transaction", lambda x: (x == "🟢 Buy").sum()),
            sells=("transaction", lambda x: (x == "🔴 Sell").sum()),
        )
        .reset_index()
    )
    return agg.sort_values("trades", ascending=False).head(top_n).reset_index(drop=True)
