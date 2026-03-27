import pandas as pd


def calculate_gex(chain):
    """
    Calculate dealer Gamma Exposure (GEX) by strike
    """

    if chain is None or chain.empty:
        return None

    if "gamma" not in chain.columns:
        return None

    chain = chain.copy()

    chain["gex"] = (
        chain["gamma"]
        * chain["openInterest"]
        * 100
        * chain["strike"]
    )

    gex_by_strike = chain.groupby("strike")["gex"].sum()

    return gex_by_strike.sort_index()