# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:53:38 2026

@author: rkafl
"""

import pandas as pd


def rank_trades(df):
    """
    Rank trade opportunities using a simple AI scoring model
    based on momentum, volatility, and volume spikes
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    df["ai_score"] = (
        abs(df["momentum"]) * 40
        + df["hv"] * 10
        + df["volume_spike"] * 20
    )

    df = df.sort_values("ai_score", ascending=False)

    return df