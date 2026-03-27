# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:52:02 2026

@author: rkafl
"""

import pandas as pd


def rank_trades(df):

    if df.empty:
        return df

    df["ai_score"] = (
        abs(df["momentum"]) * 40
        + df["hv"] * 10
        + df["volume_spike"] * 20
    )

    return df.sort_values("ai_score", ascending=False)