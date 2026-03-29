"""
build_features.py

Build a robust daily feature set from OHLCV data for next-day
range prediction.
"""

import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            col[0] if isinstance(col, tuple) and col[0] else str(col)
            for col in df.columns
        ]
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

    for lag in [1, 2, 3, 5]:
        daily[f"hl_pct_lag_{lag}"] = daily["hl_pct"].shift(lag)
        daily[f"oc_pct_lag_{lag}"] = daily["oc_pct"].shift(lag)
        daily[f"gap_pct_lag_{lag}"] = daily["gap_pct"].shift(lag)
        daily[f"close_ret_lag_{lag}"] = daily["close_ret_1d"].shift(lag)
        daily[f"volume_ratio_5_lag_{lag}"] = daily["volume_ratio_5"].shift(lag)

    daily = daily.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    daily = daily.dropna().copy()

    return daily
