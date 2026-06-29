"""
Background Data Worker
Pre-computes expensive market scans and writes results to the shared file cache
so all Streamlit users get instant loads instead of waiting for yfinance calls.

Run this as a separate process alongside the Streamlit app:
    python background_worker.py

Or schedule with Windows Task Scheduler every 15 minutes during market hours
(9:15 AM – 4:30 PM ET, Mon–Fri).

What it pre-computes every cycle:
  - Market volume leaders
  - Rapid movers
  - Strategy ideas
  - Price forecast (gainers & losers)
  - Earnings calendar (next 30 days)
  - Screener feature snapshot for all configured tickers

Cache TTL matches the refresh interval (900s = 15 min).
The dashboard reads from cache first; falling back to live fetch only when
cache is missing or expired.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

import cache_manager as cm
from config import MARKET_SCAN_TICKERS
from earnings_calendar import get_earnings_calendar
from screener_engine import load_latest_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("background_worker")

REFRESH_INTERVAL = 900  # seconds between full refresh cycles
CACHE_TTL = 960          # slightly longer than interval to cover processing time

_ET_OPEN_HOUR = 9
_ET_CLOSE_HOUR = 17


def _is_market_window() -> bool:
    """True during and around market hours (ET) Mon–Fri. Always returns True
    outside market hours so overnight runs can pre-warm for the open."""
    now = datetime.utcnow()
    # UTC-4 (EDT) or UTC-5 (EST); rough check is fine here
    et_hour = (now.hour - 4) % 24
    weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    if weekday >= 5:
        return False
    return True  # run on all weekday hours to cover pre-market and AH


def _run_volume_leaders() -> None:
    try:
        from options_data import build_market_volume_table
        log.info("Refreshing market volume leaders (%d tickers)…", len(MARKET_SCAN_TICKERS))
        df = build_market_volume_table(MARKET_SCAN_TICKERS)
        if df is not None and not df.empty:
            cm.save("market_volume", df.to_dict("records"), ttl_seconds=CACHE_TTL)
            log.info("  → saved %d rows", len(df))
    except Exception as e:
        log.warning("  volume leaders failed: %s", e)


def _run_market_movers() -> None:
    try:
        from options_data import build_market_movers_table
        log.info("Refreshing rapid movers…")
        df = build_market_movers_table(MARKET_SCAN_TICKERS)
        if df is not None and not df.empty:
            cm.save("market_movers", df.to_dict("records"), ttl_seconds=CACHE_TTL)
            log.info("  → saved %d rows", len(df))
    except Exception as e:
        log.warning("  movers failed: %s", e)


def _run_strategy() -> None:
    try:
        from options_data import build_strategy_table
        log.info("Refreshing strategy ideas…")
        df, diagnostics = build_strategy_table(MARKET_SCAN_TICKERS, top_n=10)
        if df is not None and not df.empty:
            cm.save("strategy", df.to_dict("records"), ttl_seconds=CACHE_TTL)
            cm.save("strategy_diagnostics", diagnostics, ttl_seconds=CACHE_TTL)
            log.info("  → saved %d strategy rows", len(df))
    except Exception as e:
        log.warning("  strategy failed: %s", e)


def _run_forecast() -> None:
    try:
        from options_data import build_price_forecast_table
        log.info("Refreshing price forecast…")
        gainers, losers = build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)
        if gainers is not None and not gainers.empty:
            cm.save("forecast_gainers", gainers.to_dict("records"), ttl_seconds=CACHE_TTL)
        if losers is not None and not losers.empty:
            cm.save("forecast_losers", losers.to_dict("records"), ttl_seconds=CACHE_TTL)
        log.info("  → saved forecast gainers/losers")
    except Exception as e:
        log.warning("  forecast failed: %s", e)


def _run_earnings_calendar() -> None:
    try:
        log.info("Refreshing earnings calendar…")
        df = get_earnings_calendar(MARKET_SCAN_TICKERS, max_days_ahead=30)
        if not df.empty:
            cm.save("earnings_calendar", df.to_dict("records"), ttl_seconds=CACHE_TTL * 4)
            log.info("  → saved %d upcoming earnings events", len(df))
    except Exception as e:
        log.warning("  earnings calendar failed: %s", e)


def _run_screener_snapshot() -> None:
    try:
        log.info("Refreshing screener feature snapshot (%d tickers)…", len(MARKET_SCAN_TICKERS))
        df = load_latest_features(MARKET_SCAN_TICKERS)
        if not df.empty:
            cm.save("screener_features", df.to_dict("records"), ttl_seconds=CACHE_TTL * 2)
            log.info("  → saved %d ticker feature rows", len(df))
    except Exception as e:
        log.warning("  screener snapshot failed: %s", e)


def run_one_cycle() -> None:
    log.info("=== Starting refresh cycle ===")
    start = time.time()

    _run_volume_leaders()
    _run_market_movers()
    _run_strategy()
    _run_forecast()
    _run_earnings_calendar()
    _run_screener_snapshot()

    elapsed = time.time() - start
    log.info("=== Cycle complete in %.1fs ===", elapsed)


def main() -> None:
    log.info("Background worker started. Refresh interval: %ds", REFRESH_INTERVAL)
    while True:
        if _is_market_window():
            try:
                run_one_cycle()
            except Exception as e:
                log.error("Unhandled error in cycle: %s", e)
        else:
            log.info("Outside market window (weekend), skipping cycle.")

        log.info("Sleeping %ds until next cycle…", REFRESH_INTERVAL)
        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
