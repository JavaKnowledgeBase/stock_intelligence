# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:15:32 2026

@author: rkafl
"""

from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKERS


def _first_series(data):
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def _round_or_none(value, digits=2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def analyze_ticker(ticker, period="3mo", expiry_days=30):
    df = yf.download(ticker, period=period, progress=False, threads=False)

    if df is None or df.empty:
        print(f"No data for {ticker}")
        return None

    close = _first_series(df["Close"]).dropna()
    volume = _first_series(df["Volume"]).dropna()

    if close.empty or volume.empty:
        print(f"Incomplete data for {ticker}")
        return None

    returns = close.pct_change().dropna()
    if returns.empty:
        print(f"Not enough history for {ticker}")
        return None

    hv = returns.std() * math.sqrt(252)
    momentum = close.pct_change(5).iloc[-1] if len(close) > 5 else 0.0

    avg_volume = volume.rolling(20, min_periods=5).mean().iloc[-1]
    if pd.isna(avg_volume) or avg_volume == 0:
        volume_spike = 1.0
    else:
        volume_spike = volume.iloc[-1] / avg_volume

    score = abs(momentum) * 50 + hv * 10 + volume_spike * 5

    direction = "Neutral"
    if momentum > 0.01:
        direction = "Bullish"
    elif momentum < -0.01:
        direction = "Bearish"

    price = float(close.iloc[-1])
    time_factor = math.sqrt(max(expiry_days, 1) / 252)
    expected_move_pct = max(hv * time_factor * 100, 0.5)
    expected_move_dollars = price * (expected_move_pct / 100)
    expected_low = price - expected_move_dollars
    expected_high = price + expected_move_dollars

    if direction == "Bullish":
        suggested_strike = price + expected_move_dollars * 0.5
    elif direction == "Bearish":
        suggested_strike = price - expected_move_dollars * 0.5
    else:
        suggested_strike = price

    return {
        "ticker": ticker,
        "price": _round_or_none(price),
        "momentum": _round_or_none(momentum, 3),
        "hv": _round_or_none(hv),
        "volume_spike": _round_or_none(volume_spike),
        "direction": direction,
        "score": _round_or_none(score),
        "expected_move_pct": _round_or_none(expected_move_pct),
        "expected_low": _round_or_none(expected_low),
        "expected_high": _round_or_none(expected_high),
        "suggested_strike": _round_or_none(suggested_strike),
        "period": period,
        "expiry_days": int(expiry_days),
    }


def analyze_single_ticker(ticker, period="3mo", expiry_days=30):
    return analyze_ticker(ticker, period=period, expiry_days=expiry_days)


def run_screener(period="3mo", expiry_days=30):
    results = []

    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = [
            exe.submit(analyze_ticker, ticker, period, expiry_days)
            for ticker in TICKERS
        ]

        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        print("No valid ticker data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
