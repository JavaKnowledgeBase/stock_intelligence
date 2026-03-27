# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:13:27 2026

@author: rkafl
"""

def suggest_strategy(direction, iv_rank):

    if iv_rank > 60:

        if direction == "Neutral":
            return "Iron Condor"

        if direction == "Bullish":
            return "Bull Put Spread"

        if direction == "Bearish":
            return "Bear Call Spread"

    else:

        if direction == "Bullish":
            return "Long Call"

        if direction == "Bearish":
            return "Long Put"

        return "Straddle"