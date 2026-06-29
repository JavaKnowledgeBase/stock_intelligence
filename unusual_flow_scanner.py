"""
Unusual Options Flow Scanner — finds anomalous options activity across tickers.

Signals detected:
  - Volume >> Open Interest (new money entering, not rolling)
  - High premium ($50K+ notional)
  - OTM directional bets (bullish call sweeps / bearish put sweeps)
  - Large block prints ($500K+)
"""

import concurrent.futures

import pandas as pd
import yfinance as yf


def _fetch_chain_with_spot(ticker: str, max_expirations: int = 3) -> tuple[str, float, pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        spot = getattr(t.fast_info, "last_price", None) or 0.0
        exps = t.options
        if not exps:
            return ticker, spot, pd.DataFrame()

        chains = []
        for exp in exps[:max_expirations]:
            try:
                opt = t.option_chain(exp)
                calls = opt.calls.copy()
                calls["type"] = "call"
                calls["expiration"] = exp
                puts = opt.puts.copy()
                puts["type"] = "put"
                puts["expiration"] = exp
                chains.extend([calls, puts])
            except Exception:
                continue

        if not chains:
            return ticker, spot, pd.DataFrame()

        chain = pd.concat(chains, ignore_index=True)
        chain["ticker"] = ticker
        chain["spot"] = spot
        return ticker, spot, chain
    except Exception:
        return ticker, 0.0, pd.DataFrame()


def _classify_flow(vol: float, oi: float, premium: float, vol_oi_ratio: float) -> str:
    if vol_oi_ratio > 10 and premium >= 100_000:
        return "🔥 Sweep"
    if premium >= 500_000:
        return "💰 Big Print"
    if vol_oi_ratio > 5 and vol >= 2_000:
        return "📦 Block"
    if vol_oi_ratio > 3:
        return "📈 Unusual"
    return "Activity"


def _enrich_chain(chain: pd.DataFrame) -> pd.DataFrame:
    for col in ["volume", "openInterest", "lastPrice", "impliedVolatility", "strike"]:
        chain[col] = pd.to_numeric(chain.get(col, 0), errors="coerce").fillna(0)

    spot = chain["spot"].iloc[0] if "spot" in chain.columns else 0.0

    chain["vol_oi_ratio"] = chain["volume"] / (chain["openInterest"] + 1)
    chain["premium_$"] = (chain["lastPrice"] * chain["volume"] * 100).astype(int)

    # OTM flag and distance %
    call_mask = chain["type"] == "call"
    put_mask = chain["type"] == "put"
    chain["otm"] = False
    chain["otm_pct"] = 0.0
    if spot > 0:
        chain.loc[call_mask, "otm"] = chain.loc[call_mask, "strike"] > spot
        chain.loc[put_mask, "otm"] = chain.loc[put_mask, "strike"] < spot
        chain.loc[call_mask & chain["otm"], "otm_pct"] = (
            (chain.loc[call_mask & chain["otm"], "strike"] - spot) / spot * 100
        ).round(1)
        chain.loc[put_mask & chain["otm"], "otm_pct"] = (
            (spot - chain.loc[put_mask & chain["otm"], "strike"]) / spot * 100
        ).round(1)

    chain["flow_type"] = chain.apply(
        lambda r: _classify_flow(r["volume"], r["openInterest"], r["premium_$"], r["vol_oi_ratio"]),
        axis=1,
    )

    return chain


def scan_unusual_flow(
    tickers: list[str],
    min_volume: int = 300,
    min_vol_oi_ratio: float = 2.0,
    min_premium: int = 25_000,
    max_expirations: int = 3,
    workers: int = 14,
) -> pd.DataFrame:
    """
    Scan tickers for unusual options flow.
    Returns a DataFrame sorted by premium descending, ready for display.
    """
    all_rows: list[pd.DataFrame] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_chain_with_spot, t, max_expirations): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker, spot, chain = future.result()
            if chain.empty:
                continue

            enriched = _enrich_chain(chain)

            unusual = enriched[
                (enriched["volume"] >= min_volume)
                & (enriched["vol_oi_ratio"] >= min_vol_oi_ratio)
                & (enriched["premium_$"] >= min_premium)
            ]

            if unusual.empty:
                continue

            keep = [
                "ticker", "type", "strike", "expiration",
                "volume", "openInterest", "lastPrice", "impliedVolatility",
                "vol_oi_ratio", "premium_$", "otm", "otm_pct",
                "flow_type", "spot",
            ]
            all_rows.append(unusual[[c for c in keep if c in unusual.columns]])

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sort_values("premium_$", ascending=False).reset_index(drop=True)

    # Round display columns
    df["impliedVolatility"] = (df["impliedVolatility"] * 100).round(1)
    df["vol_oi_ratio"] = df["vol_oi_ratio"].round(1)

    return df


def format_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.rename(columns={
        "ticker": "Ticker",
        "type": "C/P",
        "strike": "Strike",
        "expiration": "Expiration",
        "volume": "Volume",
        "openInterest": "Open Int",
        "lastPrice": "Last",
        "impliedVolatility": "IV %",
        "vol_oi_ratio": "Vol/OI",
        "premium_$": "Premium $",
        "otm": "OTM",
        "otm_pct": "OTM %",
        "flow_type": "Signal",
        "spot": "Spot",
    })
