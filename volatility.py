# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:12:45 2026

@author: rkafl
"""

import numpy as np


def historical_volatility(returns):

    return returns.std() * np.sqrt(252)


def iv_rank(current_iv, iv_series):

    low = np.min(iv_series)
    high = np.max(iv_series)

    if high == low:
        return 0

    return round((current_iv-low)/(high-low)*100,2)


def expected_move(price, iv, days):

    return price * iv * (days/365)**0.5