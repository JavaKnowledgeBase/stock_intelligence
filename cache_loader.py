"""
Cache Loader — pulls nightly pre-computed data from Cloudflare R2 into
Streamlit session state on app startup, eliminating all "Run" button waits.

Usage in dashboard.py (call once per session):
    import cache_loader
    meta = cache_loader.load_into_session_state(st.session_state)
"""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

log = logging.getLogger("cache_loader")

_CACHE_PREFIX = "nightly/"


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _r2_configured() -> bool:
    required = [
        "R2_BUCKET_NAME", "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
    ]
    return all(os.getenv(k) for k in required)


def _r2_get(key: str):
    """Download and parse a JSON object from R2. Returns None on any error."""
    try:
        import boto3
        client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        resp = client.get_object(
            Bucket=os.environ["R2_BUCKET_NAME"],
            Key=f"{_CACHE_PREFIX}{key}",
        )
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        log.debug("R2 get %s: %s", key, e)
        return None


# ── Public loaders ────────────────────────────────────────────────────────────

def get_cache_metadata() -> dict:
    """Build metadata: built_at, ticker_count, status per dataset."""
    return _r2_get("metadata.json") or {}


def load_price_snapshots() -> dict:
    """Dict {ticker: {spot, ret_1w, ret_1m, market_cap, ...}}."""
    return _r2_get("price_snapshots.json") or {}


def load_pc_ratios() -> pd.DataFrame:
    """DataFrame of put/call ratios sorted by P/C vol ratio desc."""
    data = _r2_get("pc_ratios.json") or {}
    if not data:
        return pd.DataFrame()
    rows = list(data.values())
    df = pd.DataFrame(rows)
    if "pc_vol_ratio" in df.columns:
        df = df.sort_values("pc_vol_ratio", ascending=False).reset_index(drop=True)
    return df


def load_unusual_flow() -> pd.DataFrame:
    """DataFrame of unusual options flow signals."""
    data = _r2_get("unusual_flow.json") or []
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_sector_rotation() -> pd.DataFrame:
    data = _r2_get("sector_rotation.json") or []
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_macro_indicators() -> dict:
    """
    Returns:
        latest     — dict  {series: {value, date}}
        yield_curve — DataFrame
        indicators  — dict {series: DataFrame with DatetimeIndex}
    """
    data = _r2_get("macro_indicators.json") or {}
    if not data:
        return {}

    # Reconstruct indicator DataFrames
    indicator_dfs: dict = {}
    for k, records in data.get("indicators", {}).items():
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        indicator_dfs[k] = df

    return {
        "latest": data.get("latest", {}),
        "yield_curve": pd.DataFrame(data.get("yield_curve", [])),
        "indicators": indicator_dfs,
    }


def load_congress_trades() -> pd.DataFrame:
    data = _r2_get("congress_trades.json") or []
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_hf_holdings() -> dict[str, dict]:
    """Dict {fund_name: {holdings: DataFrame, filing_date: str}}."""
    data = _r2_get("hf_holdings.json") or {}
    result: dict = {}
    for fund_name, fund_data in data.items():
        result[fund_name] = {
            "holdings": pd.DataFrame(fund_data.get("holdings", [])),
            "filing_date": fund_data.get("filing_date", ""),
        }
    return result


def load_earnings_briefs() -> dict:
    """Dict {ticker: {next_earnings_date, days_to_earnings, straddle, ...}}."""
    return _r2_get("earnings_briefs.json") or {}


# ── Main session-state loader ─────────────────────────────────────────────────

def load_into_session_state(ss: dict) -> dict:
    """
    Download all nightly datasets from R2 and populate Streamlit session state.

    Uses the same session state keys as each dashboard tab so the UI shows
    pre-loaded data immediately without any "Run" button interaction.

    Returns metadata dict with build info and status.
    """
    if not _r2_configured():
        log.info("R2 not configured — skipping nightly cache load")
        return {"status": "r2_not_configured"}

    meta = get_cache_metadata()
    if not meta:
        return {"status": "no_cache_found"}

    built_at = meta.get("built_at", "")
    log.info("Loading nightly cache built at %s", built_at)

    loaded: list[str] = []
    failed: list[str] = []

    def _try(name: str, loader_fn, *apply_fns):
        try:
            result = loader_fn()
            for fn in apply_fns:
                fn(result)
            loaded.append(name)
        except Exception as e:
            log.warning("Failed to load %s: %s", name, e)
            failed.append(name)

    # Price snapshots → used by AI thesis context and future ticker lookups
    def _apply_prices(data):
        if data:
            ss["nightly_price_snapshots"] = data
            ss["nightly_tickers"] = sorted(data.keys())

    _try("price_snapshots", load_price_snapshots, _apply_prices)

    # Put/Call ratios → tab13 in Options mode (pc_df / pc_fetched_at)
    def _apply_pc(df):
        if not df.empty:
            ss["pc_df"] = df
            ss["pc_fetched_at"] = built_at

    _try("pc_ratios", load_pc_ratios, _apply_pc)

    # Unusual flow → tab10 in Options mode
    def _apply_flow(df):
        if not df.empty:
            ss["unusual_flow_df"] = df
            ss["unusual_flow_fetched_at"] = built_at

    _try("unusual_flow", load_unusual_flow, _apply_flow)

    # Sector rotation → stab10 in Stocks mode
    def _apply_sector(df):
        if not df.empty:
            ss["sector_df"] = df
            ss["sector_fetched_at"] = built_at

    _try("sector_rotation", load_sector_rotation, _apply_sector)

    # Macro indicators → stab8 in Stocks mode
    def _apply_macro(macro):
        if macro:
            ss["macro_latest"] = macro.get("latest", {})
            ss["macro_yield_curve"] = macro.get("yield_curve")
            ss["macro_indicators"] = macro.get("indicators", {})
            ss["macro_fetched_at"] = built_at

    _try("macro_indicators", load_macro_indicators, _apply_macro)

    # Congress trades → tab12 in Options mode
    def _apply_congress(df):
        if not df.empty:
            ss["congress_df"] = df
            ss["congress_fetched_at"] = built_at

    _try("congress_trades", load_congress_trades, _apply_congress)

    # HF holdings → stored in ss for on-demand lookup per fund
    def _apply_hf(data):
        if data:
            ss["nightly_hf_holdings"] = data
            ss["hf_fetched_at"] = built_at

    _try("hf_holdings", load_hf_holdings, _apply_hf)

    # Earnings briefs → stored in ss for AI thesis enrichment
    def _apply_earnings(data):
        if data:
            ss["nightly_earnings_briefs"] = data

    _try("earnings_briefs", load_earnings_briefs, _apply_earnings)

    result = {
        "status": "loaded" if not failed else "partial",
        "built_at": built_at,
        "ticker_count": meta.get("ticker_count", 0),
        "loaded": loaded,
        "failed": failed,
    }
    log.info(
        "Cache load complete: %d ok, %d failed",
        len(loaded), len(failed),
    )
    return result
