# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:11:39 2026

@author: rkafl
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_DIR
from build_features import build_features

_CACHE_TTL_SECONDS = 900
_PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


def _flatten_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly flatten yfinance MultiIndex columns regardless of level order."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df = df.copy()
    for level in range(df.columns.nlevels):
        level_vals = set(df.columns.get_level_values(level))
        if _PRICE_COLS & level_vals:
            df.columns = df.columns.get_level_values(level)
            return df
    df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
    return df
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_MAX_WORKERS = 8


def _cache_key(prefix, *parts):
    return (prefix, *parts)


def _get_cached(key):
    now = time.time()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        created_at, value = entry
        if now - created_at > _CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return value


def _set_cached(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)
    return value


def _clone_frame(df):
    return df.copy() if isinstance(df, pd.DataFrame) else df


def _get_ticker(ticker):
    key = _cache_key("ticker", ticker)
    cached = _get_cached(key)
    if cached is not None:
        return cached
    return _set_cached(key, yf.Ticker(ticker))


def get_price_history(ticker, period="6mo", interval="1d"):
    key = _cache_key("history", ticker, period, interval)
    cached = _get_cached(key)
    if cached is not None:
        return _clone_frame(cached)

    history = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if history is None:
        history = pd.DataFrame()

    _set_cached(key, history.copy())
    return history


def get_stock_price(ticker):
    data = get_price_history(ticker, period="5d")
    return float(data["Close"].iloc[-1])


def get_expirations(ticker):
    key = _cache_key("expirations", ticker)
    cached = _get_cached(key)
    if cached is not None:
        return list(cached)

    tk = _get_ticker(ticker)
    expirations = list(tk.options)
    _set_cached(key, expirations)
    return expirations


def get_options_chain(ticker, expiry):
    key = _cache_key("chain", ticker, expiry)
    cached = _get_cached(key)
    if cached is not None:
        return _clone_frame(cached)

    tk = _get_ticker(ticker)
    chain = tk.option_chain(expiry)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    calls["type"] = "call"
    puts["type"] = "put"

    df = pd.concat([calls, puts], ignore_index=True)
    _set_cached(key, df.copy())
    return df


def get_market_options_snapshot(ticker, max_contracts=10):
    expirations = get_expirations(ticker)

    if not expirations:
        return pd.DataFrame()

    expiry = expirations[0]
    chain = get_options_chain(ticker, expiry)

    if chain is None or chain.empty:
        return pd.DataFrame()

    snapshot = chain.copy().rename(
        columns={
            "type": "option_type",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "inTheMoney": "in_the_money",
        }
    )

    snapshot["ticker"] = ticker
    snapshot["expiration"] = expiry

    snapshot = snapshot[
        [
            "ticker",
            "expiration",
            "option_type",
            "strike",
            "last_price",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "in_the_money",
        ]
    ]

    for column in [
        "strike",
        "last_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
    ]:
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")

    snapshot = snapshot.fillna(
        {
            "last_price": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "volume": 0,
            "open_interest": 0,
            "in_the_money": False,
        }
    )

    snapshot = snapshot.sort_values(
        ["volume", "open_interest", "last_price"],
        ascending=[False, False, False],
    )

    return snapshot.head(max_contracts).reset_index(drop=True)


def get_full_market_options_snapshot(ticker):
    expirations = get_expirations(ticker)

    if not expirations:
        return pd.DataFrame()

    expiry = expirations[0]
    chain = get_options_chain(ticker, expiry)

    if chain is None or chain.empty:
        return pd.DataFrame()

    snapshot = chain.copy().rename(
        columns={
            "type": "option_type",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "inTheMoney": "in_the_money",
        }
    )

    snapshot["ticker"] = ticker
    snapshot["expiration"] = expiry

    for column in ["volume", "open_interest", "last_price", "bid", "ask"]:
        if column in snapshot.columns:
            snapshot[column] = pd.to_numeric(
                snapshot[column],
                errors="coerce",
            ).fillna(0)

    return snapshot


def get_market_activity_snapshot_dir():
    snapshot_dir = Path(DATA_DIR) / "options_market_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _fetch_volume_row(ticker, today):
    try:
        snapshot = get_full_market_options_snapshot(ticker)
    except Exception:
        snapshot = pd.DataFrame()

    if snapshot.empty:
        return None

    call_volume = int(
        snapshot.loc[
            snapshot["option_type"] == "call",
            "volume",
        ].sum()
    )
    put_volume = int(
        snapshot.loc[
            snapshot["option_type"] == "put",
            "volume",
        ].sum()
    )
    total_volume = call_volume + put_volume
    dominant_side = "call" if call_volume >= put_volume else "put"

    return {
        "date": today.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "one_day_volume": total_volume,
        "dominant_side": dominant_side,
    }


def build_market_volume_table(tickers, lookback_days=7):
    today = pd.Timestamp.now().normalize()
    daily_rows = []

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(tickers) or 1)) as pool:
        futures = [pool.submit(_fetch_volume_row, ticker, today) for ticker in tickers]
        for future in as_completed(futures):
            row = future.result()
            if row:
                daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)

    if daily_df.empty:
        return daily_df

    snapshot_dir = get_market_activity_snapshot_dir()
    today_path = snapshot_dir / f"{today.strftime('%Y-%m-%d')}.csv"
    daily_df.to_csv(today_path, index=False)

    history_frames = []

    for csv_path in sorted(snapshot_dir.glob("*.csv")):
        try:
            history_df = pd.read_csv(csv_path)
        except Exception:
            continue

        if history_df.empty:
            continue

        history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
        history_frames.append(history_df)

    if not history_frames:
        history_df = daily_df.copy()
        history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    else:
        history_df = pd.concat(history_frames, ignore_index=True)

    cutoff = today - pd.Timedelta(days=lookback_days - 1)
    weekly_history = history_df[history_df["date"] >= cutoff].copy()

    weekly_totals = (
        weekly_history.groupby("ticker", as_index=False)["one_day_volume"]
        .sum()
        .rename(columns={"one_day_volume": "one_week_total"})
    )

    volume_pivot = (
        weekly_history.pivot_table(
            index="ticker",
            columns="date",
            values="one_day_volume",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index(axis=1)
    )

    trend_rows = []

    for ticker, row in volume_pivot.iterrows():
        series = row.tolist()
        non_zero_series = [value for value in series if value > 0]

        if len(non_zero_series) < 2:
            trend = "insufficient history"
        elif len(series) >= 4:
            split_index = max(1, len(series) // 2)
            earlier_avg = sum(series[:split_index]) / split_index
            recent_avg = sum(series[split_index:]) / max(1, len(series[split_index:]))

            if recent_avg > earlier_avg * 1.05:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.95:
                trend = "decreasing"
            else:
                trend = "flat"
        else:
            trend = "increasing" if series[-1] > series[0] else "decreasing"

        trend_rows.append(
            {
                "ticker": ticker,
                "one_week_trend": trend,
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    scanned_total = int(daily_df["one_day_volume"].sum())

    result = daily_df.merge(weekly_totals, on="ticker", how="left")
    result = result.merge(trend_df, on="ticker", how="left")
    result["percent_of_day_total"] = (
        result["one_day_volume"] / scanned_total * 100
        if scanned_total > 0 else 0
    )

    result = result.sort_values(
        ["one_day_volume", "one_week_total"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


def _get_spy_ret_5d():
    """Cached 5-day SPY return for relative strength calculation."""
    try:
        spy = _flatten_df_columns(get_price_history("SPY", period="1mo", interval="1d"))
        close = pd.to_numeric(spy["Close"], errors="coerce").dropna()
        if len(close) < 6:
            return 0.0
        return float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100)
    except Exception:
        return 0.0


def _build_mover_row(ticker):
    try:
        return _build_mover_row_inner(ticker)
    except Exception:
        return None


def _build_mover_row_inner(ticker):
    try:
        history = get_price_history(ticker, period="1y", interval="1d")
    except Exception:
        history = pd.DataFrame()

    if history is None or history.empty:
        return None

    features = build_features(history)

    if features is None or features.empty:
        return None

    latest = features.iloc[-1]
    close_price = float(pd.to_numeric(latest["Close"], errors="coerce"))
    ret_5d = float(pd.to_numeric(latest["close_ret_5d"], errors="coerce"))
    ret_3d = float(pd.to_numeric(latest["close_ret_3d"], errors="coerce"))
    vol_5d = float(pd.to_numeric(latest["volatility_5d"], errors="coerce"))
    vol_10d = float(pd.to_numeric(latest["volatility_10d"], errors="coerce"))
    volume_ratio_5 = float(pd.to_numeric(latest["volume_ratio_5"], errors="coerce"))
    volume_ratio_20 = float(pd.to_numeric(latest["volume_ratio_20"], errors="coerce"))
    dist_ma_20 = float(pd.to_numeric(latest["dist_ma_20_pct"], errors="coerce"))
    dist_ema_20 = float(pd.to_numeric(latest["dist_ema_20_pct"], errors="coerce"))
    rsi_14 = float(pd.to_numeric(latest.get("rsi_14", 50), errors="coerce"))
    macd_hist = float(pd.to_numeric(latest.get("macd_hist", 0), errors="coerce"))
    adx_14 = float(pd.to_numeric(latest.get("adx_14", 0), errors="coerce"))
    _ma_align_raw = pd.to_numeric(latest.get("ma_alignment", 0), errors="coerce")
    ma_alignment = int(_ma_align_raw) if pd.notna(_ma_align_raw) else 0
    trend_consistency = float(pd.to_numeric(latest.get("trend_consistency_10d", 0.5), errors="coerce"))
    vol_dir_ratio = float(pd.to_numeric(latest.get("vol_direction_ratio", 1.0), errors="coerce"))
    atr_14 = float(pd.to_numeric(latest.get("atr_14", 0), errors="coerce"))
    obv_slope = float(pd.to_numeric(latest.get("obv_slope_10d", 0), errors="coerce"))
    pct_from_52w_high = float(pd.to_numeric(latest.get("pct_from_52w_high", -50), errors="coerce"))

    # Relative strength vs SPY (ticker 5d return minus SPY 5d return)
    spy_ret_5d = _get_spy_ret_5d()
    rel_strength = ret_5d - spy_ret_5d

    # 30-day annualized historical volatility
    try:
        _flat = _flatten_df_columns(history.copy())
        _close = pd.to_numeric(_flat["Close"], errors="coerce").dropna()
        _log_ret = np.log(_close / _close.shift(1)).dropna()
        hv_30d = float(_log_ret.tail(30).std() * np.sqrt(252) * 100)
    except Exception:
        hv_30d = 0.0

    swing_factor = (vol_5d * 0.6) + (vol_10d * 0.4)
    volume_factor = (volume_ratio_5 * 0.6) + (volume_ratio_20 * 0.4)
    trend_factor = (ret_5d * 0.65) + (ret_3d * 0.35)
    extension_factor = (dist_ma_20 * 0.6) + (dist_ema_20 * 0.4)

    # RSI additive adjustment (calls: RSI 50-70 ideal; puts: RSI 30-50 ideal)
    rsi_adj_up = 2.0 if 50 <= rsi_14 <= 70 else (-3.0 if rsi_14 > 75 else 0.0)
    rsi_adj_dn = 2.0 if 30 <= rsi_14 <= 50 else (-3.0 if rsi_14 < 25 else 0.0)

    # MACD histogram additive adjustment
    macd_adj = 1.5 if macd_hist > 0 else -1.5

    # ADX trend strength: weak trend penalizes both sides; strong trend rewards aligned side
    adx_strength = -2.0 if adx_14 < 20 else (2.5 if adx_14 > 40 else (2.0 if adx_14 >= 25 else 0.0))

    # MA alignment: +1 bullish stack boosts upside, -1 bearish stack boosts downside
    ma_adj_up = 2.0 if ma_alignment == 1 else (-2.0 if ma_alignment == -1 else 0.0)
    ma_adj_dn = -ma_adj_up

    # Trend consistency: fraction of last 10 days with positive returns
    tc_adj_up = 1.5 if trend_consistency > 0.6 else (-1.5 if trend_consistency < 0.4 else 0.0)
    tc_adj_dn = -tc_adj_up

    # Volume direction ratio: up-day vs down-day volume
    vd_adj_up = 1.0 if vol_dir_ratio > 1.2 else (-1.0 if vol_dir_ratio < 0.8 else 0.0)
    vd_adj_dn = -vd_adj_up

    # OBV slope: accumulation (+) vs distribution (-)
    obv_adj_up = 0.75 if obv_slope > 5 else (-0.75 if obv_slope < -5 else 0.0)
    obv_adj_dn = -obv_adj_up

    # 52-week high proximity: near highs = breakout potential; far from highs = laggard
    high52_adj_up = 1.0 if pct_from_52w_high >= -3 else (-0.5 if pct_from_52w_high < -20 else 0.0)
    high52_adj_dn = 0.5 if pct_from_52w_high < -20 else (-1.0 if pct_from_52w_high >= -3 else 0.0)

    # Relative strength vs SPY
    rs_adj_up = 1.0 if rel_strength > 2 else (-1.0 if rel_strength < -2 else 0.0)
    rs_adj_dn = -rs_adj_up

    one_week_upside = (
        trend_factor * 0.9
        + swing_factor * 0.8
        + max(volume_factor - 1, 0) * 6
        + max(extension_factor, 0) * 0.35
        + rsi_adj_up + macd_adj + adx_strength + ma_adj_up + tc_adj_up + vd_adj_up
        + obv_adj_up + high52_adj_up + rs_adj_up
    )
    one_week_downside = (
        (-trend_factor) * 0.9
        + swing_factor * 0.8
        + max(volume_factor - 1, 0) * 6
        + max(-extension_factor, 0) * 0.35
        + rsi_adj_dn + (-macd_adj) + adx_strength + ma_adj_dn + tc_adj_dn + vd_adj_dn
        + obv_adj_dn + high52_adj_dn + rs_adj_dn
    )

    one_month_upside = (
        trend_factor * 1.1
        + swing_factor * 0.9
        + max(volume_factor - 1, 0) * 8
        + max(extension_factor, 0) * 0.45
        + rsi_adj_up + macd_adj + adx_strength + ma_adj_up + tc_adj_up + vd_adj_up
        + obv_adj_up + high52_adj_up + rs_adj_up
    )
    one_month_downside = (
        (-trend_factor) * 1.1
        + swing_factor * 0.9
        + max(volume_factor - 1, 0) * 8
        + max(-extension_factor, 0) * 0.45
        + rsi_adj_dn + (-macd_adj) + adx_strength + ma_adj_dn + tc_adj_dn + vd_adj_dn
        + obv_adj_dn + high52_adj_dn + rs_adj_dn
    )

    direction_1w = "Grow Rapidly" if one_week_upside >= one_week_downside else "Fall Steeply"
    direction_1m = "Grow Rapidly" if one_month_upside >= one_month_downside else "Fall Steeply"

    one_week_score = max(one_week_upside, one_week_downside)
    one_month_score = max(one_month_upside, one_month_downside)

    return {
        "ticker": ticker,
        "close": round(close_price, 2),
        "one_week_view": direction_1w,
        "one_week_score": round(one_week_score, 2),
        "one_month_view": direction_1m,
        "one_month_score": round(one_month_score, 2),
        "ret_5d": round(ret_5d, 2),
        "volatility_5d": round(vol_5d, 2),
        "volume_ratio_5": round(volume_ratio_5, 2),
        "rsi_14": round(rsi_14, 1),
        "adx_14": round(adx_14, 1),
        "atr_14": round(atr_14, 3),
        "rel_strength_vs_spy": round(rel_strength, 2),
        "pct_from_52w_high": round(pct_from_52w_high, 2),
        "obv_slope_10d": round(obv_slope, 2),
        "hv_30d": round(hv_30d, 2),
    }


def build_market_movers_table(tickers):
    rows = []

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(tickers) or 1)) as pool:
        futures = [pool.submit(_build_mover_row, ticker) for ticker in tickers]
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)

    movers_df = pd.DataFrame(rows)

    if movers_df.empty:
        return movers_df

    movers_df = movers_df.sort_values(
        ["one_week_score", "one_month_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return movers_df


def build_price_forecast_table(tickers, top_n=10):
    """
    Return two DataFrames (gainers, losers) each with top_n tickers,
    estimated % move for next week and next 2 weeks derived from
    technical momentum scores.
    """
    movers_df = build_market_movers_table(tickers)

    if movers_df is None or movers_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = movers_df.copy()
    df = df.dropna(subset=["one_week_score", "one_month_score", "volatility_5d"]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2-week score: blend of 1-week and 1-month
    df["two_week_score"] = df["one_week_score"].astype(float) * 0.6 + df["one_month_score"].astype(float) * 0.4

    # Convert scores to % estimates, capped by each ticker's own volatility range
    vol_cap = (df["volatility_5d"].astype(float) * 5).clip(lower=2.0).values

    df["est_1w_pct"] = np.minimum(df["one_week_score"].astype(float) * 0.35, vol_cap)
    df["est_2w_pct"] = np.minimum(df["two_week_score"].astype(float) * 0.45, vol_cap * 1.4)

    # Apply direction sign
    up_mask_1w = df["one_week_view"] == "Grow Rapidly"
    up_mask_2w = df["one_month_view"] == "Grow Rapidly"

    df["est_1w_pct"] = df["est_1w_pct"].where(up_mask_1w, -df["est_1w_pct"])
    df["est_2w_pct"] = df["est_2w_pct"].where(up_mask_2w, -df["est_2w_pct"])

    df["est_1w_pct"] = df["est_1w_pct"].round(2)
    df["est_2w_pct"] = df["est_2w_pct"].round(2)

    display_cols = [
        "ticker", "close", "one_week_view", "est_1w_pct",
        "est_2w_pct", "rsi_14", "adx_14", "volatility_5d", "volume_ratio_5",
    ]

    gainers = (
        df[df["est_1w_pct"] > 0]
        .sort_values("est_1w_pct", ascending=False)
        .head(top_n)[display_cols]
        .reset_index(drop=True)
    )
    losers = (
        df[df["est_1w_pct"] < 0]
        .sort_values("est_1w_pct", ascending=True)
        .head(top_n)[display_cols]
        .reset_index(drop=True)
    )

    return gainers, losers


def _get_expiration_near_date(ticker, target_date):
    """Return the available expiration closest to target_date."""
    expirations = get_expirations(ticker)
    if not expirations:
        return None
    target = pd.Timestamp(target_date).normalize()
    best, best_diff = None, None
    for expiry in expirations:
        expiry_dt = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_dt):
            continue
        diff = abs((expiry_dt.normalize() - target).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = expiry
    return best


def _upcoming_fridays(n=2):
    """Return the next n Fridays from today as Timestamps."""
    today = pd.Timestamp.now().normalize()
    fridays = []
    d = today
    while len(fridays) < n:
        d += pd.Timedelta(days=1)
        if d.dayofweek == 4:  # Friday
            fridays.append(d)
    return fridays


def _get_strategy_expiration(ticker, min_days=21, max_days=45):
    expirations = get_expirations(ticker)

    if not expirations:
        return None

    today = pd.Timestamp.now().normalize()
    dated_expirations = []

    for expiry in expirations:
        expiry_date = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_date):
            continue

        days_to_expiry = int((expiry_date.normalize() - today).days)
        dated_expirations.append((expiry, days_to_expiry))

    preferred = [
        expiry for expiry, days in dated_expirations
        if min_days <= days <= max_days
    ]

    if preferred:
        return preferred[0]

    future_expiries = [
        (expiry, days) for expiry, days in dated_expirations if days >= 7
    ]

    if future_expiries:
        future_expiries.sort(key=lambda item: abs(item[1] - 30))
        return future_expiries[0][0]

    return dated_expirations[0][0] if dated_expirations else None


def _pick_strategy_contract(ticker, contract_type, close_price, hv_30d=0.0, expiry=None):
    if expiry is None:
        expiry = _get_strategy_expiration(ticker)

    if expiry is None:
        return None

    chain = get_options_chain(ticker, expiry)

    if chain is None or chain.empty:
        return None

    chain = chain.copy().rename(
        columns={
            "type": "option_type",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
        }
    )

    filtered = chain[chain["option_type"] == contract_type].copy()

    if filtered.empty:
        return None

    for column in ["strike", "last_price", "bid", "ask", "volume", "open_interest"]:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce").fillna(0)

    filtered["mid_price"] = (filtered["bid"] + filtered["ask"]) / 2
    filtered["spread_pct"] = (
        (filtered["ask"] - filtered["bid"]) / filtered["mid_price"].replace(0, pd.NA) * 100
    )

    base_filter = (
        (filtered["ask"] > 0)
        & (filtered["bid"] >= 0)
        & (filtered["mid_price"] > 0)
    )
    filtered = filtered[base_filter].copy()

    if filtered.empty:
        return None

    # Two-pass liquidity filter: tight first, fallback to loose
    tight = filtered[
        (filtered["volume"] >= 100)
        & (filtered["open_interest"] >= 100)
        & (filtered["spread_pct"] <= 12)
    ]
    working = tight if not tight.empty else filtered[
        (filtered["volume"] >= 50)
        & (filtered["open_interest"] >= 100)
        & (filtered["spread_pct"] <= 18)
    ]

    if working.empty:
        return None

    target_multiplier = 1.02 if contract_type == "call" else 0.98
    target_strike = close_price * target_multiplier

    working = working.copy()
    working["strike_distance"] = (working["strike"] - target_strike).abs()
    working["distance_pct"] = working["strike_distance"] / max(close_price, 1) * 100
    working["liquidity_score"] = working["volume"] + (working["open_interest"] * 0.5)
    max_liq = working["liquidity_score"].max()
    working["contract_quality_score"] = (
        (working["liquidity_score"] / max_liq) * 55
        + ((18 - working["spread_pct"]).clip(lower=0) / 18) * 30
        + ((10 - working["distance_pct"]).clip(lower=0) / 10) * 15
    )

    working = working.sort_values(
        ["contract_quality_score", "liquidity_score", "last_price"],
        ascending=[False, False, False],
    )

    selected = working.iloc[0]

    # IV / HV ratio
    iv_hv_ratio = None
    try:
        iv_raw = selected.get("impliedVolatility", None) if hasattr(selected, "get") else selected["impliedVolatility"] if "impliedVolatility" in selected.index else None
        if iv_raw is not None and pd.notna(iv_raw) and hv_30d > 0:
            iv_annualized = float(iv_raw) * 100
            iv_hv_ratio = round(iv_annualized / hv_30d, 2)
    except Exception:
        iv_hv_ratio = None

    used_tight = not tight.empty

    return {
        "expiration": expiry,
        "contract_type": contract_type,
        "strike": round(float(selected["strike"]), 2),
        "option_value": round(float(selected["last_price"]), 2),
        "bid": round(float(selected["bid"]), 2),
        "ask": round(float(selected["ask"]), 2),
        "spread_pct": round(float(selected["spread_pct"]), 2),
        "open_interest": int(selected["open_interest"]),
        "volume": int(selected["volume"]),
        "contract_quality_score": round(float(selected["contract_quality_score"]), 2),
        "iv_hv_ratio": iv_hv_ratio,
        "tight_liquidity": used_tight,
    }


def _build_strategy_recommendation(row):
    hv_30d = float(row.get("hv_30d", 0) or 0)
    expiry_override = row.get("_target_expiry") or None
    horizon = row.get("_horizon", "")
    contract = _pick_strategy_contract(
        row["ticker"],
        row["contract_type"],
        row["close"],
        hv_30d=hv_30d,
        expiry=expiry_override,
    )

    if not contract:
        return None

    ticker = row["ticker"]
    underlying_move_pct = abs(float(row["ret_5d"]))
    contract_side = contract["contract_type"]
    rsi = float(row.get("rsi_14", 50) or 50)
    adx = float(row.get("adx_14", 0) or 0)

    mid_price = (contract["bid"] + contract["ask"]) / 2 if contract["ask"] > 0 else contract["option_value"]
    mid_price = round(mid_price, 2)

    # ATR-based stop/target sizing: use ticker ATR scaled to option price
    atr_14 = float(row.get("atr_14", 0) or 0)
    close_price = float(row.get("close", 1) or 1)
    atr_pct = (atr_14 / close_price) if close_price > 0 else 0.02
    # Stop = 1.5× ATR from entry; targets = 2× and 3× ATR, bounded to 15%–35%
    stop_mult = max(0.15, min(0.30, atr_pct * 1.5))
    t1_mult = max(0.20, min(0.35, atr_pct * 2.0))
    t2_mult = max(0.30, min(0.50, atr_pct * 3.0))
    stop_price = round(mid_price * (1 - stop_mult), 2)
    target_1 = round(mid_price * (1 + t1_mult), 2)
    target_2 = round(mid_price * (1 + t2_mult), 2)
    stop_pct = round(stop_mult * 100)
    t1_pct = round(t1_mult * 100)
    t2_pct = round(t2_mult * 100)

    entry_rule = (
        f"Enter near open if {ticker} confirms direction. "
        f"Target entry around ${mid_price:.2f} mid-price."
    )
    if rsi > 68:
        entry_rule += f" RSI elevated ({rsi:.0f}) — wait for a small pullback before entering."
    elif rsi < 32:
        entry_rule += f" RSI oversold ({rsi:.0f}) — wait for stabilization before entering."

    stop_rule = (
        f"Stop loss at ${stop_price:.2f} (−{stop_pct}% from entry, 1.5× ATR). "
        f"Exit immediately if {ticker} reverses the morning trend."
    )
    take_profit_rule = (
        f"Take 50% off at ${target_1:.2f} (+{t1_pct}%, 2× ATR). "
        f"Trail remainder to ${target_2:.2f} (+{t2_pct}%, 3× ATR) if momentum holds."
    )
    midday_rule = (
        f"At 11 AM check {ticker} volume vs open. "
        f"If volume is fading and price is stalling near ${mid_price:.2f}, exit remaining position."
    )
    daily_plan = (
        f"Bias {contract_side} on morning confirmation. "
        f"Risk ${stop_price:.2f}, first target ${target_1:.2f}, extended target ${target_2:.2f}."
    )

    return {
        "ticker": ticker,
        "horizon": horizon,
        "view": row["signal_direction"],
        "contract_type": contract["contract_type"],
        "expiration": contract["expiration"],
        "strike_price": contract["strike"],
        "option_value": contract["option_value"],
        "bid": contract["bid"],
        "ask": contract["ask"],
        "spread_pct": contract["spread_pct"],
        "open_interest": contract["open_interest"],
        "option_volume": contract["volume"],
        "contract_quality_score": contract["contract_quality_score"],
        "iv_hv_ratio": contract.get("iv_hv_ratio"),
        "rsi_14": round(rsi, 1),
        "adx_14": round(adx, 1),
        "strategy_score": round(float(row["strategy_score"]), 2),
        "day_volume_share": round(float(row["percent_of_day_total"]), 2),
        "underlying_5d_move": round(underlying_move_pct, 2),
        "entry_rule": entry_rule,
        "stop_rule": stop_rule,
        "take_profit_rule": take_profit_rule,
        "midday_check": midday_rule,
        "daily_plan": daily_plan,
    }


def _get_market_regime():
    try:
        spy = get_price_history("SPY", period="3mo", interval="1d")
        if spy is None or spy.empty:
            return "neutral"
        spy = _flatten_df_columns(spy)
        close = pd.to_numeric(spy["Close"], errors="coerce").dropna()
        if len(close) < 20:
            return "neutral"
        ret_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100
        ma_20 = close.tail(20).mean()
        above_ma20 = close.iloc[-1] > ma_20
        if ret_5d > 1.0 and above_ma20:
            return "bullish"
        elif ret_5d < -1.0 and not above_ma20:
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def build_strategy_table(tickers, top_n=10):
    movers_df = build_market_movers_table(tickers)
    volume_df = build_market_volume_table(tickers)
    regime = _get_market_regime()

    diagnostics = {
        "tickers_requested": len(tickers),
        "movers_found": 0 if movers_df is None else len(movers_df),
        "volume_rows_found": 0 if volume_df is None else len(volume_df),
        "combined_candidates": 0,
        "contracts_evaluated": 0,
        "contracts_selected": 0,
        "regime": regime,
        "status": "ok",
        "message": "",
    }

    if movers_df.empty or volume_df.empty:
        diagnostics["status"] = "data_unavailable"
        diagnostics["message"] = (
            "Yahoo data was incomplete or unavailable for the latest run."
        )
        return pd.DataFrame(), diagnostics

    combined = movers_df.merge(
        volume_df[
            [
                "ticker",
                "dominant_side",
                "one_day_volume",
                "percent_of_day_total",
                "call_volume",
                "put_volume",
            ]
        ],
        on="ticker",
        how="left",
    )
    diagnostics["combined_candidates"] = len(combined)

    use_month_view = combined["one_month_score"] >= combined["one_week_score"]
    combined["signal_direction"] = combined["one_week_view"]
    combined.loc[use_month_view, "signal_direction"] = combined.loc[use_month_view, "one_month_view"]
    combined["contract_type"] = combined["signal_direction"].map(
        {
            "Grow Rapidly": "call",
            "Fall Steeply": "put",
        }
    )
    combined["strategy_score"] = (
        combined["one_week_score"] * 0.35
        + combined["one_month_score"] * 0.45
        + combined["percent_of_day_total"].fillna(0) * 0.20
    )
    combined["signal_strength"] = combined[["one_week_score", "one_month_score"]].max(axis=1)

    # Regime filter: suppress weak trades that fight the broad market
    if regime in ("bullish", "bearish") and len(combined) > 0:
        score_threshold = combined["strategy_score"].quantile(0.60)
        aligned_side = "call" if regime == "bullish" else "put"
        combined = combined[
            (combined["contract_type"] == aligned_side)
            | (combined["strategy_score"] >= score_threshold)
        ]

    combined = combined.sort_values(
        ["strategy_score", "signal_strength", "one_day_volume"],
        ascending=[False, False, False],
    )

    # Select top_n unique tickers, then expand each into horizon rows
    today = pd.Timestamp.now().normalize()
    spy_horizons = [
        (today + pd.Timedelta(days=5), "+5d"),
        (today + pd.Timedelta(days=10), "+10d"),
        (today + pd.Timedelta(days=15), "+15d"),
        (today + pd.Timedelta(days=21), "+3wk"),
    ]
    fridays = _upcoming_fridays(2)
    friday_labels = ["Next Fri", "Fri+2"]

    seen_tickers = set()
    expanded_rows = []
    seen_expiry_keys = set()

    for _, row in combined.iterrows():
        ticker = row["ticker"]
        if ticker not in seen_tickers:
            seen_tickers.add(ticker)
        if len(seen_tickers) > top_n:
            break

        if ticker == "SPY":
            horizons = spy_horizons
            labels = [label for _, label in spy_horizons]
            targets = [target for target, _ in spy_horizons]
        else:
            horizons = list(zip(fridays, friday_labels))
            targets = fridays
            labels = friday_labels

        for target, label in zip(targets, labels):
            expiry = _get_expiration_near_date(ticker, target)
            if not expiry:
                continue
            key = (ticker, expiry)
            if key in seen_expiry_keys:
                continue
            seen_expiry_keys.add(key)
            new_row = row.copy()
            new_row["_target_expiry"] = expiry
            new_row["_horizon"] = label
            expanded_rows.append(new_row)

    candidates = pd.DataFrame(expanded_rows) if expanded_rows else pd.DataFrame()

    recommendations = []
    max_workers = min(8, len(candidates) or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_build_strategy_recommendation, row)
            for _, row in candidates.iterrows()
        ]

        diagnostics["contracts_evaluated"] = len(futures)

        for future in as_completed(futures):
            recommendation = future.result()
            if recommendation:
                recommendations.append(recommendation)

    if recommendations:
        # Sort: SPY first (market context), then by strategy score
        recommendations = sorted(
            recommendations,
            key=lambda item: (
                0 if item["ticker"] == "SPY" else 1,
                -item["strategy_score"],
                -item["contract_quality_score"],
            ),
        )

    diagnostics["contracts_selected"] = len(recommendations)

    if not recommendations:
        diagnostics["status"] = "no_contracts"
        diagnostics["message"] = (
            "Signals were found, but no option contracts passed the liquidity and spread filters."
        )
    elif len(recommendations) < top_n:
        diagnostics["status"] = "partial_results"
        diagnostics["message"] = (
            "Only part of the target list was available because some chains were missing or filtered out."
        )
    else:
        diagnostics["message"] = "Strategy ideas generated successfully."

    return pd.DataFrame(recommendations), diagnostics
