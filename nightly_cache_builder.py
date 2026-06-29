"""
Nightly Cache Builder — pre-computes all platform data for ~1000 tickers
and uploads to Cloudflare R2 so the dashboard loads instantly at startup.

Run via GitHub Actions cron at 2 AM UTC (10 PM ET) on weeknights:
    python nightly_cache_builder.py

Requires env vars: R2_BUCKET_NAME, R2_ENDPOINT_URL,
                   R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nightly_builder")

CACHE_PREFIX = "nightly/"
WORKERS = 30


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


class _Encoder(json.JSONEncoder):
    """Handle pandas Timestamps and numpy scalars for JSON serialization."""
    def default(self, o):
        if isinstance(o, (pd.Timestamp, datetime)):
            return str(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def r2_put(key: str, data) -> None:
    """Upload JSON-serialisable data to R2 under the nightly prefix."""
    body = json.dumps(data, cls=_Encoder).encode("utf-8")
    _r2_client().put_object(
        Bucket=os.environ["R2_BUCKET_NAME"],
        Key=f"{CACHE_PREFIX}{key}",
        Body=body,
        ContentType="application/json",
    )
    log.info("  → r2_put %s (%d bytes)", key, len(body))


# ── Ticker universe ───────────────────────────────────────────────────────────

def get_universe() -> list[str]:
    """
    Build ~1000-ticker universe: MARKET_SCAN_TICKERS + S&P 500 from Wikipedia.
    Wikipedia is the authoritative, free, live-updating source for S&P 500 members.
    """
    from config import MARKET_SCAN_TICKERS
    tickers: list[str] = list(MARKET_SCAN_TICKERS)

    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        sp500 = tables[0]["Symbol"].tolist()
        # Wikipedia uses dots (BRK.B), yfinance uses dashes (BRK-B)
        sp500 = [t.replace(".", "-") for t in sp500]
        before = len(tickers)
        tickers = list(dict.fromkeys(tickers + sp500))
        log.info(
            "Universe: %d base + %d S&P 500 = %d unique tickers",
            before, len(sp500), len(tickers),
        )
    except Exception as e:
        log.warning("Wikipedia S&P 500 fetch failed: %s", e)

    return tickers[:1000]


# ── Data builders ─────────────────────────────────────────────────────────────

def build_price_snapshots(tickers: list[str]) -> dict:
    """
    Fast-info + 1-month history for every ticker.
    Returns {ticker: {spot, prev_close, ret_1w, ret_1m, market_cap, ...}}
    """
    def _fetch(ticker: str) -> tuple[str, dict | None]:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            spot = getattr(fi, "last_price", None) or 0.0
            prev = getattr(fi, "previous_close", None) or 0.0

            hist = t.history(period="1mo", auto_adjust=True)
            ret_1w = ret_1m = avg_vol = None
            if not hist.empty:
                c = hist["Close"]
                if len(c) >= 5:
                    ret_1w = round((c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] * 100, 2)
                if len(c) >= 20:
                    ret_1m = round((c.iloc[-1] - c.iloc[0]) / c.iloc[0] * 100, 2)
                avg_vol = int(hist["Volume"].mean())

            return ticker, {
                "spot": round(float(spot), 2),
                "prev_close": round(float(prev), 2),
                "ret_1w": ret_1w,
                "ret_1m": ret_1m,
                "avg_vol_30d": avg_vol,
                "market_cap": int(getattr(fi, "market_cap", 0) or 0),
                "52w_high": round(float(getattr(fi, "fifty_two_week_high", 0) or 0), 2),
                "52w_low": round(float(getattr(fi, "fifty_two_week_low", 0) or 0), 2),
                "shares_outstanding": int(getattr(fi, "shares", 0) or 0),
            }
        except Exception:
            return ticker, None

    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch, t): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            ticker, data = fut.result()
            if data:
                results[ticker] = data
            if (i + 1) % 100 == 0:
                log.info("  price snapshots: %d/%d done (%d ok)", i + 1, len(tickers), len(results))

    return results


def build_pc_ratios(tickers: list[str]) -> dict:
    """
    Put/call ratios for all optionable tickers.
    Returns {ticker: {pc_vol_ratio, pc_oi_ratio, sentiment, ...}}
    """
    from putcall_scanner import _fetch_pc_ratio

    results: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_pc_ratio, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            if row:
                results[row["ticker"]] = row

    return results


def build_unusual_flow(tickers: list[str]) -> list:
    """
    Unusual options flow scan across all tickers.
    Returns list of flow records sorted by premium.
    """
    from unusual_flow_scanner import scan_unusual_flow
    df = scan_unusual_flow(
        tickers,
        min_volume=500,
        min_vol_oi_ratio=2.0,
        min_premium=50_000,
        workers=WORKERS,
    )
    return df.to_dict("records") if not df.empty else []


def build_sector_rotation() -> list:
    from sector_rotation import get_sector_rotation
    df = get_sector_rotation(workers=11)
    return df.to_dict("records") if not df.empty else []


def build_macro_indicators() -> dict:
    from macro_dashboard import (
        get_macro_indicators,
        get_latest_values,
        get_yield_curve,
    )

    indicators = get_macro_indicators(years_back=2)
    latest = get_latest_values(indicators)
    _, yield_curve = get_yield_curve(months_back=3)

    # Serialise DataFrames to records; convert DatetimeIndex to string column
    serialised: dict = {}
    for k, df in indicators.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df2 = df.copy()
            if hasattr(df2.index, "strftime"):
                df2.index = df2.index.strftime("%Y-%m-%d")
            df2.index.name = "date"
            serialised[k] = df2.reset_index().to_dict("records")

    # latest values may contain non-JSON types from FRED
    def _clean(v):
        if isinstance(v, dict):
            return {k2: _clean(v2) for k2, v2 in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, pd.Timestamp):
            return str(v)
        return v

    return {
        "indicators": serialised,
        "latest": _clean(latest),
        "yield_curve": (
            yield_curve.to_dict("records") if not yield_curve.empty else []
        ),
    }


def build_congress_trades() -> list:
    from congress_tracker import get_congress_trades
    df = get_congress_trades(days_back=90)
    return df.to_dict("records") if not df.empty else []


def build_hf_holdings(fund_limit: int = 10) -> dict:
    """
    Fetch 13F holdings for top N funds (rate-limited; ~0.12s per SEC request).
    Returns {fund_name: {holdings: [...], filing_date: str}}
    """
    from hedge_fund_tracker import KNOWN_FUNDS, get_fund_holdings

    results: dict = {}
    for fund_name in list(KNOWN_FUNDS.keys())[:fund_limit]:
        try:
            df, filing_date, _ = get_fund_holdings(fund_name)
            if not df.empty:
                results[fund_name] = {
                    "holdings": df.head(100).to_dict("records"),
                    "filing_date": str(filing_date),
                }
                log.info("  HF %s: %d holdings as of %s", fund_name, len(df), filing_date)
            time.sleep(0.5)  # extra courtesy sleep between funds
        except Exception as e:
            log.warning("  HF %s failed: %s", fund_name, e)

    return results


def build_earnings_briefs(tickers: list[str], limit: int = 300) -> dict:
    """
    Earnings analytics for the top N liquid tickers.
    Skips DataFrame fields (those are per-user on-demand) to keep payload small.
    """
    from earnings_analyzer import analyse_earnings

    priority = tickers[:limit]
    results: dict = {}

    for i, ticker in enumerate(priority):
        try:
            raw = analyse_earnings(ticker)
            # Strip non-serialisable DataFrames
            raw.pop("earnings_history", None)
            raw.pop("post_moves", None)
            raw.pop("strategy_tips", None)  # list[str] — keep if present
            results[ticker] = raw
            if (i + 1) % 50 == 0:
                log.info("  earnings briefs: %d/%d done", i + 1, len(priority))
        except Exception as e:
            log.warning("  earnings brief %s: %s", ticker, e)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    log.info("=== Nightly Cache Build Started ===")
    start = time.time()
    built_at = datetime.now(timezone.utc).isoformat()
    status: dict = {}

    # Validate R2 credentials before doing any work
    required = ["R2_BUCKET_NAME", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing R2 env vars: {', '.join(missing)}")

    tickers = get_universe()
    log.info("Working with %d tickers", len(tickers))

    # ── 1. Price snapshots (all tickers, parallel) ────────────────────────────
    log.info("[1/8] Price snapshots for %d tickers…", len(tickers))
    try:
        prices = build_price_snapshots(tickers)
        r2_put("price_snapshots.json", prices)
        status["price_snapshots"] = len(prices)
    except Exception as e:
        log.error("Price snapshots failed: %s", e)
        status["price_snapshots"] = f"ERROR: {e}"

    # ── 2. Put/Call ratios (optionable tickers, parallel) ─────────────────────
    log.info("[2/8] Put/call ratios…")
    try:
        pc = build_pc_ratios(tickers)
        r2_put("pc_ratios.json", pc)
        status["pc_ratios"] = len(pc)
    except Exception as e:
        log.error("PC ratios failed: %s", e)
        status["pc_ratios"] = f"ERROR: {e}"

    # ── 3. Unusual options flow scan ──────────────────────────────────────────
    log.info("[3/8] Unusual flow scan…")
    try:
        flow = build_unusual_flow(tickers)
        r2_put("unusual_flow.json", flow)
        status["unusual_flow"] = len(flow)
    except Exception as e:
        log.error("Unusual flow failed: %s", e)
        status["unusual_flow"] = f"ERROR: {e}"

    # ── 4. Sector rotation (11 ETFs) ──────────────────────────────────────────
    log.info("[4/8] Sector rotation…")
    try:
        sectors = build_sector_rotation()
        r2_put("sector_rotation.json", sectors)
        status["sector_rotation"] = len(sectors)
    except Exception as e:
        log.error("Sector rotation failed: %s", e)
        status["sector_rotation"] = f"ERROR: {e}"

    # ── 5. Macro indicators (FRED + Treasury) ─────────────────────────────────
    log.info("[5/8] Macro indicators…")
    try:
        macro = build_macro_indicators()
        r2_put("macro_indicators.json", macro)
        status["macro_indicators"] = "ok"
    except Exception as e:
        log.error("Macro indicators failed: %s", e)
        status["macro_indicators"] = f"ERROR: {e}"

    # ── 6. Congress trades (House + Senate stock watcher) ─────────────────────
    log.info("[6/8] Congress trades…")
    try:
        congress = build_congress_trades()
        r2_put("congress_trades.json", congress)
        status["congress_trades"] = len(congress)
    except Exception as e:
        log.error("Congress trades failed: %s", e)
        status["congress_trades"] = f"ERROR: {e}"

    # ── 7. Hedge fund 13F holdings (SEC EDGAR, rate-limited) ──────────────────
    log.info("[7/8] Hedge fund holdings…")
    try:
        hf = build_hf_holdings(fund_limit=10)
        r2_put("hf_holdings.json", hf)
        status["hf_holdings"] = len(hf)
    except Exception as e:
        log.error("HF holdings failed: %s", e)
        status["hf_holdings"] = f"ERROR: {e}"

    # ── 8. Earnings briefs (top 300 liquid tickers) ───────────────────────────
    log.info("[8/8] Earnings briefs for top 300 tickers…")
    try:
        earnings = build_earnings_briefs(tickers, limit=300)
        r2_put("earnings_briefs.json", earnings)
        status["earnings_briefs"] = len(earnings)
    except Exception as e:
        log.error("Earnings briefs failed: %s", e)
        status["earnings_briefs"] = f"ERROR: {e}"

    # ── Metadata ──────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start)
    metadata = {
        "built_at": built_at,
        "build_duration_seconds": elapsed,
        "ticker_count": len(tickers),
        "tickers": tickers,
        "status": status,
    }
    r2_put("metadata.json", metadata)

    log.info("=== Build complete in %ds ===", elapsed)
    for k, v in status.items():
        log.info("  %-22s %s", k, v)

    return metadata


if __name__ == "__main__":
    main()
