"""
download_intraday.py

Downloads daily OHLCV price data for a single ticker.
"""

import yfinance as yf
import pandas as pd


def download_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., 'NVDA')
    period : str
        Lookback period (default: '2y')

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by datetime
    """

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return None

    # Ensure proper datetime index
    df.index = pd.to_datetime(df.index)

    return df
