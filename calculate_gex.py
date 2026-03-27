# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:47:11 2026

@author: rkafl
"""

import pandas as pd

def calculate_gex(chain):

    if "gamma" not in chain.columns:
        return None

    chain["gex"] = (
        chain["gamma"]
        * chain["openInterest"]
        * 100
        * chain["strike"]
    )

    gex_by_strike = chain.groupby("strike")["gex"].sum()

    return gex_by_strike.sort_index()