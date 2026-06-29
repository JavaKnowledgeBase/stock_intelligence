"""
Multi-Factor Market Screener
Uses pre-computed daily features from data/features/ for near-instant screening.
Falls back to live yfinance for any ticker without a feature file.

Five built-in presets cover the most common trader setups:
  Oversold Bounce | Breakout Setup | High Momentum |
  Short Squeeze Risk | Pullback in Uptrend
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from build_features import build_features
from config import DATA_DIR

_FEATURE_DIR = Path(DATA_DIR) / "features"

# ── Preset definitions ────────────────────────────────────────────────────────

SCREENER_PRESETS: dict[str, dict] = {
    "Oversold Bounce": {
        "description": (
            "RSI < 35 · stock down >20% from 52W high · volume spike (5d vs 20d avg > 1.5×). "
            "Mean-reversion candidates where selling may be exhausted."
        ),
        "filters": {
            "rsi_14":            {"max": 35},
            "pct_from_52w_high": {"max": -20},
            "volume_ratio_5":    {"min": 1.5},
        },
    },
    "Breakout Setup": {
        "description": (
            "RSI 50–65 · price above 20-day MA · volume confirming (ratio > 1.2×) · "
            "bullish MA stack. Momentum just starting to build."
        ),
        "filters": {
            "rsi_14":          {"min": 50, "max": 65},
            "dist_ma_20_pct":  {"min": 0},
            "volume_ratio_5":  {"min": 1.2},
            "ma_alignment":    {"min": 0},
        },
    },
    "High Momentum": {
        "description": (
            "5D return > 5% · volume 1.5× above 20-day avg · fully bullish MA stack. "
            "Strong trend continuation candidates."
        ),
        "filters": {
            "close_ret_5d":    {"min": 5},
            "volume_ratio_20": {"min": 1.5},
            "ma_alignment":    {"min": 1},
        },
    },
    "Short Squeeze Risk": {
        "description": (
            "RSI > 60 · volume > 2× average · within 10% of 52W high. "
            "Stocks that could squeeze higher on continued buying pressure."
        ),
        "filters": {
            "rsi_14":            {"min": 60},
            "volume_ratio_5":    {"min": 2.0},
            "pct_from_52w_high": {"min": -10},
        },
    },
    "Pullback in Uptrend": {
        "description": (
            "Bullish MA stack · RSI 40–55 · price within 5% below 20-day MA. "
            "Healthy trend with a buyable dip."
        ),
        "filters": {
            "ma_alignment":   {"min": 1},
            "rsi_14":         {"min": 40, "max": 55},
            "dist_ma_20_pct": {"min": -5, "max": 2},
        },
    },
}

# ── Displayable filter columns ────────────────────────────────────────────────

FILTER_COLUMNS: dict[str, str] = {
    "rsi_14":                 "RSI (14)",
    "close_ret_5d":           "5D Return %",
    "close_ret_3d":           "3D Return %",
    "volume_ratio_5":         "Vol Ratio 5d",
    "volume_ratio_20":        "Vol Ratio 20d",
    "dist_ma_20_pct":         "Dist from 20MA %",
    "pct_from_52w_high":      "% from 52W High",
    "pct_from_52w_low":       "% from 52W Low",
    "adx_14":                 "ADX (14)",
    "bb_pct":                 "BB Position %",
    "ma_alignment":           "MA Alignment",
    "trend_consistency_10d":  "Trend Consistency",
    "volatility_5d":          "Volatility 5d",
    "obv_slope_10d":          "OBV Slope 10d",
    "macd_hist":              "MACD Histogram",
    "atr_14":                 "ATR (14)",
}


def _load_from_file(ticker: str) -> Optional[pd.Series]:
    fpath = _FEATURE_DIR / f"{ticker}.csv"
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if df.empty:
            return None
        row = df.iloc[-1].copy()
        row["ticker"] = ticker
        row["close"] = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
        return row
    except Exception:
        return None


def _fetch_live(ticker: str) -> Optional[pd.Series]:
    try:
        import data_fetcher as _df
        hist = _df.get(ticker, period="6mo")
        if hist is None or hist.empty or len(hist) < 40:
            hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if hist is None or hist.empty or len(hist) < 40:
            return None
        feats = build_features(hist)
        if feats.empty:
            return None
        row = feats.iloc[-1].copy()
        row["ticker"] = ticker
        row["close"] = float(hist["Close"].iloc[-1])
        return row
    except Exception:
        return None


def load_latest_features(tickers: list[str], live_workers: int = 50) -> pd.DataFrame:
    """
    Load the most recent feature row for each ticker.
    Tries local CSV first; fetches live for any missing.
    """
    rows: list[pd.Series] = []
    missing: list[str] = []

    for ticker in tickers:
        row = _load_from_file(ticker)
        if row is not None:
            rows.append(row)
        else:
            missing.append(ticker)

    if missing:
        with ThreadPoolExecutor(max_workers=live_workers) as pool:
            futs = {pool.submit(_fetch_live, t): t for t in missing}
            for f in as_completed(futs, timeout=90):
                res = f.result()
                if res is not None:
                    rows.append(res)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).reset_index(drop=True)
    # Ensure ticker column is first
    cols = ["ticker", "close"] + [c for c in df.columns if c not in ("ticker", "close")]
    df = df[[c for c in cols if c in df.columns]]
    return df


def apply_filters(df: pd.DataFrame, filters: dict[str, dict]) -> pd.DataFrame:
    if df.empty or not filters:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for col, bounds in filters.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if "min" in bounds:
            mask &= vals >= bounds["min"]
        if "max" in bounds:
            mask &= vals <= bounds["max"]
    return df[mask].copy()


def run_screener(
    tickers: list[str],
    filters: dict[str, dict],
    sort_by: str = "rsi_14",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Screen tickers by applying filters to their latest feature values.
    Returns a display-ready DataFrame with the most relevant columns.
    """
    import data_fetcher as _df
    _df.bulk_preload(tickers, period="1y")
    df = load_latest_features(tickers)
    if df.empty:
        return df

    result = apply_filters(df, filters)
    if result.empty:
        return result

    display_cols = ["ticker", "close"] + [c for c in FILTER_COLUMNS if c in result.columns]
    available = [c for c in display_cols if c in result.columns]
    result = result[available].copy()

    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending)

    return result.round(2).reset_index(drop=True)
