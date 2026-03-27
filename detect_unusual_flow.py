# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:49:14 2026

@author: rkafl
"""

import pandas as pd


def detect_unusual_flow(chain):

    if chain.empty:
        return chain

    chain["volume_oi_ratio"] = chain["volume"] / (chain["openInterest"] + 1)

    flow = chain[
        (chain["volume"] > 500)
        & (chain["volume_oi_ratio"] > 2)
    ]

    return flow.sort_values("volume", ascending=False)