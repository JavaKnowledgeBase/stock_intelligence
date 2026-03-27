import pandas as pd


def detect_unusual_flow(chain):
    """
    Detect unusual options activity based on
    high volume relative to open interest
    """

    if chain is None or chain.empty:
        return pd.DataFrame()

    chain = chain.copy()

    # avoid divide-by-zero
    chain["volume_oi_ratio"] = chain["volume"] / (chain["openInterest"] + 1)

    unusual = chain[
        (chain["volume"] > 500) &
        (chain["volume_oi_ratio"] > 2)
    ]

    unusual = unusual.sort_values("volume", ascending=False)

    return unusual