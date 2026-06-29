"""
Sector Rotation Tracker — monitors money flows across S&P 500 sector ETFs.

Source: yfinance (free, no API key)
Sectors: XLK XLF XLE XLV XLI XLP XLU XLB XLRE XLC XLY
"""

import concurrent.futures

import pandas as pd
import yfinance as yf

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLC": "Communication",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}


def _fetch_sector(etf: str, name: str) -> dict | None:
    try:
        t = yf.Ticker(etf)
        hist = t.history(period="6mo", auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None

        c = hist["Close"]
        price = c.iloc[-1]

        def _ret(n):
            return round((c.iloc[-1] - c.iloc[-min(n, len(c))]) / c.iloc[-min(n, len(c))] * 100, 2)

        ret_1w = _ret(5)
        ret_1m = _ret(21)
        ret_3m = _ret(63)

        # Volume trend: recent 5d vs 20d avg
        v = hist["Volume"]
        vol_ratio = round(v.iloc[-5:].mean() / v.iloc[-20:].mean(), 2) if len(v) >= 20 else 1.0

        # Realised vol
        rets = c.pct_change().dropna()
        rv_30 = round(rets.iloc[-30:].std() * (252 ** 0.5) * 100, 1) if len(rets) >= 30 else None

        # Momentum score: simple weighted combo
        momentum = round(ret_1w * 0.4 + ret_1m * 0.35 + ret_3m * 0.25, 2)

        return {
            "etf": etf,
            "sector": name,
            "price": round(price, 2),
            "ret_1w": ret_1w,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "vol_ratio": vol_ratio,
            "rv_30d": rv_30,
            "momentum_score": momentum,
        }
    except Exception:
        return None


def get_sector_rotation(workers: int = 11) -> pd.DataFrame:
    """
    Fetch performance data for all sector ETFs.
    Returns DataFrame sorted by 1-month return descending (rotating into first).
    """
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_sector, etf, name): etf
                   for etf, name in SECTOR_ETFS.items()}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("ret_1m", ascending=False).reset_index(drop=True)

    # Rank: 1 = best momentum
    df["momentum_rank"] = df["momentum_score"].rank(ascending=False).astype(int)

    # Label: rotating in / out
    median = df["ret_1m"].median()
    df["flow_label"] = df["ret_1m"].apply(
        lambda r: "🟢 Rotating In" if r > median + 1 else
                  "🔴 Rotating Out" if r < median - 1 else
                  "⚪ Neutral"
    )
    return df


def get_rotation_summary(df: pd.DataFrame) -> dict:
    """Key rotation signals from the sector data."""
    if df.empty:
        return {}

    top3 = df.head(3)
    bot3 = df.tail(3)

    # Risk-on vs risk-off: compare cyclicals vs defensives
    cyclicals = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLE"]
    defensives = ["XLP", "XLU", "XLV", "XLRE", "XLB"]

    cyc = df[df["etf"].isin(cyclicals)]["ret_1m"].mean()
    def_ = df[df["etf"].isin(defensives)]["ret_1m"].mean()

    if cyc > def_ + 1:
        risk_regime = "🟢 Risk-On — cyclicals outperforming defensives"
    elif def_ > cyc + 1:
        risk_regime = "🔴 Risk-Off — defensives outperforming cyclicals"
    else:
        risk_regime = "⚪ Mixed — no clear risk regime"

    return {
        "risk_regime": risk_regime,
        "leading_sectors": ", ".join(top3["sector"].tolist()),
        "lagging_sectors": ", ".join(bot3["sector"].tolist()),
        "cyclical_avg_1m": round(cyc, 2),
        "defensive_avg_1m": round(def_, 2),
    }
