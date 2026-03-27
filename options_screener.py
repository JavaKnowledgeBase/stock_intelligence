# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:15:32 2026

@author: rkafl
"""

import yfinance as yf
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from config import TICKERS


def analyze_ticker(ticker):

    df = yf.download(ticker, period="3mo", progress=False, threads=False)

    if df is None or df.empty:
        print(f"No data for {ticker}")
        return None

    close = df["Close"]
    volume = df["Volume"]

    # Fix multi-index columns
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    returns = close.pct_change()

    hv = returns.std() * np.sqrt(252)

    momentum = close.pct_change(5).iloc[-1]

    volume_spike = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

    score = abs(momentum) * 50 + hv * 10 + volume_spike * 5

    direction = "Neutral"

    if momentum > 0.01:
        direction = "Bullish"

    elif momentum < -0.01:
        direction = "Bearish"

    return {
        "ticker": ticker,
        "momentum": round(momentum, 3),
        "hv": round(hv, 2),
        "volume_spike": round(volume_spike, 2),
        "direction": direction,
        "score": round(score, 2)
    }


def run_screener():

    results = []

    with ThreadPoolExecutor(max_workers=10) as exe:

        futures = [exe.submit(analyze_ticker, t) for t in TICKERS]

        for f in futures:

            r = f.result()

            if r is not None:
                results.append(r)

    # After threads finish
    if len(results) == 0:
        print("No valid ticker data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df.sort_values("score", ascending=False)