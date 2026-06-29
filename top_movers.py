"""
Top Movers — today's biggest gainers and losers from the ticker universe.
Uses yfinance fast_info for lightweight per-ticker % change.
"""

import concurrent.futures

import pandas as pd


def _fetch_ticker_move(ticker: str) -> dict | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = fi.last_price
        prev = fi.previous_close
        if not price or not prev or prev == 0:
            return None
        change_pct = (price - prev) / prev * 100
        return {
            "ticker": ticker,
            "price": round(price, 2),
            "prev_close": round(prev, 2),
            "change_pct": round(change_pct, 2),
            "volume": getattr(fi, "three_month_average_volume", None),
        }
    except Exception:
        return None


def get_top_movers(
    tickers: list[str],
    top_n: int = 15,
    workers: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (gainers_df, losers_df) each with top_n rows sorted by abs % change.
    """
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_ticker_move, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["change_pct"])

    gainers = df[df["change_pct"] > 0].nlargest(top_n, "change_pct").reset_index(drop=True)
    losers = df[df["change_pct"] < 0].nsmallest(top_n, "change_pct").reset_index(drop=True)

    return gainers, losers
