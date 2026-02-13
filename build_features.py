"""
build_features.py

Builds daily ML features from OHLCV price data.
This module is SAFE after CSV reloads and resistant
to datetime index corruption.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds technical features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns:
            Open, High, Low, Close, Volume
        Index:
            Date or datetime (or convertible to datetime)

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features and no NaNs.
    """

    # -------------------------------------------------
    # STEP 1: Normalize index to DatetimeIndex
    # -------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop rows where datetime conversion failed
    df = df[~df.index.isna()]

    # -------------------------------------------------
    # STEP 2: Date-based features
    # -------------------------------------------------
    df["date"] = df.index.date
    df["day_of_week"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month

    # -------------------------------------------------
    # STEP 3: Price-based features
    # -------------------------------------------------
    df["hl_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["oc_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100

    # -------------------------------------------------
    # STEP 4: Rolling statistics
    # -------------------------------------------------
    df["volatility_5d"] = df["hl_pct"].rolling(window=5).mean()
    df["volatility_10d"] = df["hl_pct"].rolling(window=10).mean()

    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # -------------------------------------------------
    # STEP 5: Cleanup
    # -------------------------------------------------
    df.dropna(inplace=True)

    return df
