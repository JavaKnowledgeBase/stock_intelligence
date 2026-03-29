"""
build_features.py

Build a robust daily feature set from OHLCV data for next-day
range prediction.
"""

import pandas as pd
import numpy as np


_PRICE_COLS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df = df.copy()
    # Find whichever MultiIndex level contains the actual price column names
    for level in range(df.columns.nlevels):
        level_vals = set(df.columns.get_level_values(level))
        if _PRICE_COLS & level_vals:
            df.columns = df.columns.get_level_values(level)
            return df
    # Fallback: use level 0
    df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
        if col in df.columns:
            series_or_frame = df[col]
            if isinstance(series_or_frame, pd.DataFrame):
                series_or_frame = series_or_frame.iloc[:, 0]
            df[col] = pd.to_numeric(series_or_frame, errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features from daily OHLCV data.
    """
    try:
        return _build_features_inner(df)
    except Exception:
        return pd.DataFrame()


def _build_features_inner(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_columns(df.copy())
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _coerce_numeric_columns(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if len(df) < 40:
        return pd.DataFrame()

    daily = df.copy()

    # Force all OHLCV to float64 to prevent any object-dtype arithmetic
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        daily[col] = daily[col].astype(float)

    price_range = (daily["High"] - daily["Low"]).replace(0, pd.NA)
    prev_close = daily["Close"].shift(1).replace(0, pd.NA)

    daily["date"] = daily.index.strftime("%Y-%m-%d")
    daily["day_of_week"] = daily.index.dayofweek
    daily["day"] = daily.index.day
    daily["month"] = daily.index.month
    daily["quarter"] = daily.index.quarter
    daily["is_month_start"] = daily.index.is_month_start.astype(int)
    daily["is_month_end"] = daily.index.is_month_end.astype(int)

    daily["hl_pct"] = (daily["High"] - daily["Low"]) / daily["Close"] * 100
    daily["oc_pct"] = (daily["Close"] - daily["Open"]) / daily["Open"] * 100
    daily["gap_pct"] = (daily["Open"] - prev_close) / prev_close * 100
    daily["close_ret_1d"] = daily["Close"].pct_change(1) * 100
    daily["close_ret_3d"] = daily["Close"].pct_change(3) * 100
    daily["close_ret_5d"] = daily["Close"].pct_change(5) * 100
    daily["volume_ret_1d"] = daily["Volume"].pct_change(1) * 100

    daily["intraday_position"] = (daily["Close"] - daily["Low"]) / price_range
    daily["upper_shadow_pct"] = (daily["High"] - daily[["Open", "Close"]].max(axis=1)) / daily["Close"] * 100
    daily["lower_shadow_pct"] = (daily[["Open", "Close"]].min(axis=1) - daily["Low"]) / daily["Close"] * 100

    daily["volatility_5d"] = daily["hl_pct"].rolling(window=5).mean()
    daily["volatility_10d"] = daily["hl_pct"].rolling(window=10).mean()
    daily["volatility_std_5d"] = daily["hl_pct"].rolling(window=5).std()
    daily["volatility_std_10d"] = daily["hl_pct"].rolling(window=10).std()

    daily["ma_5"] = daily["Close"].rolling(window=5).mean()
    daily["ma_10"] = daily["Close"].rolling(window=10).mean()
    daily["ma_20"] = daily["Close"].rolling(window=20).mean()
    daily["ema_5"] = daily["Close"].ewm(span=5, adjust=False).mean()
    daily["ema_10"] = daily["Close"].ewm(span=10, adjust=False).mean()
    daily["ema_20"] = daily["Close"].ewm(span=20, adjust=False).mean()

    daily["dist_ma_5_pct"] = (daily["Close"] - daily["ma_5"]) / daily["Close"] * 100
    daily["dist_ma_10_pct"] = (daily["Close"] - daily["ma_10"]) / daily["Close"] * 100
    daily["dist_ma_20_pct"] = (daily["Close"] - daily["ma_20"]) / daily["Close"] * 100
    daily["dist_ema_10_pct"] = (daily["Close"] - daily["ema_10"]) / daily["Close"] * 100
    daily["dist_ema_20_pct"] = (daily["Close"] - daily["ema_20"]) / daily["Close"] * 100

    daily["volume_ratio_5"] = daily["Volume"] / daily["Volume"].rolling(window=5).mean()
    daily["volume_ratio_20"] = daily["Volume"] / daily["Volume"].rolling(window=20).mean()

    # RSI (14)
    _delta = daily["Close"].diff()
    _gain = _delta.clip(lower=0)
    _loss = (-_delta).clip(lower=0)
    _avg_gain = _gain.ewm(com=13, adjust=False).mean()
    _avg_loss = _loss.ewm(com=13, adjust=False).mean()
    _rs = _avg_gain / _avg_loss.replace(0, pd.NA)
    daily["rsi_14"] = (100 - (100 / (1 + _rs))).clip(0, 100)

    # MACD (12, 26, 9)
    _ema12 = daily["Close"].ewm(span=12, adjust=False).mean()
    _ema26 = daily["Close"].ewm(span=26, adjust=False).mean()
    daily["macd"] = _ema12 - _ema26
    daily["macd_signal"] = daily["macd"].ewm(span=9, adjust=False).mean()
    daily["macd_hist"] = daily["macd"] - daily["macd_signal"]

    # Bollinger Band position (20, 2)
    _bb_mid = daily["Close"].rolling(20).mean()
    _bb_std = daily["Close"].rolling(20).std()
    _bb_upper = _bb_mid + 2 * _bb_std
    _bb_lower = _bb_mid - 2 * _bb_std
    _bb_range = (_bb_upper - _bb_lower).replace(0, pd.NA)
    daily["bb_pct"] = (daily["Close"] - _bb_lower) / _bb_range

    # ADX (14) — trend strength, direction-neutral
    _prev_high = daily["High"].shift(1)
    _prev_low = daily["Low"].shift(1)
    _prev_close = daily["Close"].shift(1)
    _tr = pd.concat([
        daily["High"] - daily["Low"],
        (daily["High"] - _prev_close).abs(),
        (daily["Low"] - _prev_close).abs(),
    ], axis=1).max(axis=1)
    _dm_plus_raw = (daily["High"] - _prev_high).astype(float)
    _dm_minus_raw = (_prev_low - daily["Low"]).astype(float)
    _plus_dm = _dm_plus_raw.where(
        (_dm_plus_raw > _dm_minus_raw) & (_dm_plus_raw > 0), 0.0
    )
    _minus_dm = _dm_minus_raw.where(
        (_dm_minus_raw > _dm_plus_raw) & (_dm_minus_raw > 0), 0.0
    )
    _atr14 = _tr.ewm(alpha=1 / 14, adjust=False).mean()
    daily["atr_14"] = _atr14
    _plus_di14 = 100 * _plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / _atr14.replace(0, pd.NA)
    _minus_di14 = 100 * _minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / _atr14.replace(0, pd.NA)
    _di_sum = (_plus_di14 + _minus_di14).replace(0, pd.NA)
    _dx = 100 * (_plus_di14 - _minus_di14).abs() / _di_sum
    daily["adx_14"] = _dx.ewm(alpha=1 / 14, adjust=False).mean()
    daily["plus_di_14"] = _plus_di14
    daily["minus_di_14"] = _minus_di14

    # MA alignment: +1 bullish stack, -1 bearish stack, 0 mixed
    daily["ma_alignment"] = (
        ((daily["ma_5"] > daily["ma_10"]) & (daily["ma_10"] > daily["ma_20"])).astype(int)
        - ((daily["ma_5"] < daily["ma_10"]) & (daily["ma_10"] < daily["ma_20"])).astype(int)
    )

    # Trend consistency: fraction of last 10 days with positive close return
    _ret_pos = (daily["close_ret_1d"] > 0).astype(float)
    daily["trend_consistency_10d"] = _ret_pos.rolling(10).mean()

    # Volume direction ratio: up-day volume vs down-day volume over last 10 days
    _up_vol = (daily["Volume"] * (daily["close_ret_1d"] > 0).astype(float)).rolling(10).sum()
    _dn_vol = (daily["Volume"] * (daily["close_ret_1d"] <= 0).astype(float)).rolling(10).sum()
    daily["vol_direction_ratio"] = _up_vol / _dn_vol.where(_dn_vol > 0)

    # OBV and OBV slope (10-day % change)
    _close_diff = daily["Close"].diff()
    _obv_direction = ((_close_diff > 0).astype(float) - (_close_diff < 0).astype(float))
    _obv = (daily["Volume"] * _obv_direction).cumsum()
    daily["obv"] = _obv
    _obv_prev = _obv.shift(10)
    _obv_prev_safe = _obv_prev.where(_obv_prev.abs() > 0)
    daily["obv_slope_10d"] = (_obv - _obv_prev) / _obv_prev_safe.abs() * 100

    # 52-week high/low proximity (% from rolling high/low using available history)
    _lookback = min(252, len(daily))
    daily["high_52w"] = daily["High"].rolling(window=_lookback, min_periods=20).max()
    daily["low_52w"] = daily["Low"].rolling(window=_lookback, min_periods=20).min()
    daily["pct_from_52w_high"] = (daily["Close"] - daily["high_52w"]) / daily["high_52w"] * 100
    daily["pct_from_52w_low"] = (daily["Close"] - daily["low_52w"]) / daily["low_52w"] * 100

    for lag in [1, 2, 3, 5]:
        daily[f"hl_pct_lag_{lag}"] = daily["hl_pct"].shift(lag)
        daily[f"oc_pct_lag_{lag}"] = daily["oc_pct"].shift(lag)
        daily[f"gap_pct_lag_{lag}"] = daily["gap_pct"].shift(lag)
        daily[f"close_ret_lag_{lag}"] = daily["close_ret_1d"].shift(lag)
        daily[f"volume_ratio_5_lag_{lag}"] = daily["volume_ratio_5"].shift(lag)

    daily = daily.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    daily = daily.dropna().copy()

    return daily
