# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:11:39 2026

@author: rkafl
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

import pandas as pd
import yfinance as yf

from config import DATA_DIR
from build_features import build_features

_CACHE_TTL_SECONDS = 900
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


def _build_mover_row(ticker):
    try:
        history = get_price_history(ticker, period="6mo", interval="1d")
    except Exception:
        history = pd.DataFrame()

    if history is None or history.empty:
        return None

    features = build_features(history)

    if features.empty:
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

    swing_factor = ((vol_5d * 0.6) + (vol_10d * 0.4))
    volume_factor = ((volume_ratio_5 * 0.6) + (volume_ratio_20 * 0.4))
    trend_factor = ((ret_5d * 0.65) + (ret_3d * 0.35))
    extension_factor = ((dist_ma_20 * 0.6) + (dist_ema_20 * 0.4))

    one_week_upside = (
        trend_factor * 0.9
        + swing_factor * 0.8
        + max(volume_factor - 1, 0) * 6
        + max(extension_factor, 0) * 0.35
    )
    one_week_downside = (
        (-trend_factor) * 0.9
        + swing_factor * 0.8
        + max(volume_factor - 1, 0) * 6
        + max(-extension_factor, 0) * 0.35
    )

    one_month_upside = (
        trend_factor * 1.1
        + swing_factor * 0.9
        + max(volume_factor - 1, 0) * 8
        + max(extension_factor, 0) * 0.45
    )
    one_month_downside = (
        (-trend_factor) * 1.1
        + swing_factor * 0.9
        + max(volume_factor - 1, 0) * 8
        + max(-extension_factor, 0) * 0.45
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


def _pick_strategy_contract(ticker, contract_type, close_price):
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

    filtered = filtered[
        (filtered["ask"] > 0)
        & (filtered["bid"] >= 0)
        & (filtered["volume"] >= 50)
        & (filtered["open_interest"] >= 100)
    ].copy()

    if filtered.empty:
        return None

    filtered["mid_price"] = (filtered["bid"] + filtered["ask"]) / 2
    filtered = filtered[filtered["mid_price"] > 0].copy()

    if filtered.empty:
        return None

    filtered["spread_pct"] = (
        (filtered["ask"] - filtered["bid"]) / filtered["mid_price"] * 100
    )
    filtered = filtered[filtered["spread_pct"] <= 18].copy()

    if filtered.empty:
        return None

    target_multiplier = 1.02 if contract_type == "call" else 0.98
    target_strike = close_price * target_multiplier

    filtered["strike_distance"] = (filtered["strike"] - target_strike).abs()
    filtered["distance_pct"] = filtered["strike_distance"] / max(close_price, 1) * 100
    filtered["liquidity_score"] = filtered["volume"] + (filtered["open_interest"] * 0.5)
    filtered["contract_quality_score"] = (
        (filtered["liquidity_score"] / filtered["liquidity_score"].max()) * 55
        + ((18 - filtered["spread_pct"]).clip(lower=0) / 18) * 30
        + ((10 - filtered["distance_pct"]).clip(lower=0) / 10) * 15
    )

    filtered = filtered.sort_values(
        ["contract_quality_score", "liquidity_score", "last_price"],
        ascending=[False, False, False],
    )

    selected = filtered.iloc[0]

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
    }


def _build_strategy_recommendation(row):
    contract = _pick_strategy_contract(
        row["ticker"],
        row["contract_type"],
        row["close"],
    )

    if not contract:
        return None

    underlying_move_pct = abs(float(row["ret_5d"]))
    contract_side = contract["contract_type"]
    entry_rule = (
        "Enter only if price confirms the opening move and option spread stays tight."
    )
    stop_rule = (
        "Exit if premium drops 20% from entry or if the underlying reverses the morning trend."
    )
    take_profit_rule = (
        "Take profit into 15%-25% premium gains and trail the rest only if momentum remains strong."
    )
    midday_rule = (
        "Recheck near 11 AM; avoid new entries if volume fades and price stalls."
    )

    return {
        "ticker": row["ticker"],
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
        "strategy_score": round(float(row["strategy_score"]), 2),
        "day_volume_share": round(float(row["percent_of_day_total"]), 2),
        "underlying_5d_move": round(underlying_move_pct, 2),
        "entry_rule": entry_rule,
        "stop_rule": stop_rule,
        "take_profit_rule": take_profit_rule,
        "midday_check": midday_rule,
        "daily_plan": (
            f"Bias {contract_side} on morning confirmation, then reassess at 11 AM."
        ),
    }


def build_strategy_table(tickers, top_n=10):
    movers_df = build_market_movers_table(tickers)
    volume_df = build_market_volume_table(tickers)

    diagnostics = {
        "tickers_requested": len(tickers),
        "movers_found": 0 if movers_df is None else len(movers_df),
        "volume_rows_found": 0 if volume_df is None else len(volume_df),
        "combined_candidates": 0,
        "contracts_evaluated": 0,
        "contracts_selected": 0,
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

    combined = combined.sort_values(
        ["strategy_score", "signal_strength", "one_day_volume"],
        ascending=[False, False, False],
    )

    candidate_limit = min(len(combined), max(top_n * 3, top_n + 4, 12))
    candidates = combined.head(candidate_limit).copy()

    recommendations = []
    max_workers = min(6, len(candidates) or 1)

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
        recommendations = sorted(
            recommendations,
            key=lambda item: (item["strategy_score"], item["contract_quality_score"]),
            reverse=True,
        )[:top_n]

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
