"""
Put/Call Ratio Scanner — aggregate options sentiment across the ticker universe.

High P/C ratio (>1.5) = bearish sentiment / potential contrarian buy
Low P/C ratio (<0.5) = bullish sentiment / potential contrarian sell / complacency

Source: yfinance options chain (free, no API key)
"""

import concurrent.futures

import pandas as pd
import yfinance as yf


def _fetch_pc_ratio(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        # Use nearest 1-2 expirations for current sentiment
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        atm_call_iv = None
        atm_put_iv = None

        spot = getattr(t.fast_info, "last_price", None) or 0.0

        for exp in exps[:2]:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts

                calls["volume"] = pd.to_numeric(calls["volume"], errors="coerce").fillna(0)
                puts["volume"] = pd.to_numeric(puts["volume"], errors="coerce").fillna(0)
                calls["openInterest"] = pd.to_numeric(calls["openInterest"], errors="coerce").fillna(0)
                puts["openInterest"] = pd.to_numeric(puts["openInterest"], errors="coerce").fillna(0)

                total_call_vol += calls["volume"].sum()
                total_put_vol += puts["volume"].sum()
                total_call_oi += calls["openInterest"].sum()
                total_put_oi += puts["openInterest"].sum()

                # ATM IV for skew
                if spot > 0 and atm_call_iv is None:
                    atm_idx = (calls["strike"] - spot).abs().idxmin()
                    atm_strike = calls.loc[atm_idx, "strike"]
                    call_row = calls[calls["strike"] == atm_strike]
                    put_row = puts[puts["strike"] == atm_strike]
                    if not call_row.empty:
                        atm_call_iv = call_row["impliedVolatility"].iloc[0] * 100
                    if not put_row.empty:
                        atm_put_iv = put_row["impliedVolatility"].iloc[0] * 100
            except Exception:
                continue

        if total_call_vol + total_put_vol == 0:
            return None

        pc_vol = round(total_put_vol / (total_call_vol + 1), 3)
        pc_oi = round(total_put_oi / (total_call_oi + 1), 3)
        skew = round((atm_put_iv or 0) - (atm_call_iv or 0), 1)

        # Sentiment label from P/C volume ratio
        if pc_vol >= 1.5:
            sentiment = "🐻 Very Bearish"
        elif pc_vol >= 1.0:
            sentiment = "🔴 Bearish"
        elif pc_vol <= 0.4:
            sentiment = "🐂 Very Bullish"
        elif pc_vol <= 0.7:
            sentiment = "🟢 Bullish"
        else:
            sentiment = "⚪ Neutral"

        return {
            "ticker": ticker,
            "spot": round(spot, 2),
            "call_vol": int(total_call_vol),
            "put_vol": int(total_put_vol),
            "pc_vol_ratio": pc_vol,
            "call_oi": int(total_call_oi),
            "put_oi": int(total_put_oi),
            "pc_oi_ratio": pc_oi,
            "iv_skew": skew,
            "sentiment": sentiment,
        }
    except Exception:
        return None


def scan_putcall_ratios(
    tickers: list[str],
    workers: int = 16,
) -> pd.DataFrame:
    """
    Scan put/call ratios across all tickers.
    Returns DataFrame sorted by P/C volume ratio descending (most bearish first).
    """
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_pc_ratio, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("pc_vol_ratio", ascending=False).reset_index(drop=True)
    return df


def get_extreme_readings(df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (most_bearish, most_bullish) — contrarian signal candidates.
    Most bearish (high P/C) = potential oversold / contrarian long.
    Most bullish (low P/C) = potential complacency / contrarian short.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    most_bearish = df.head(top_n).copy()
    most_bullish = df.tail(top_n).sort_values("pc_vol_ratio").copy()
    return most_bearish, most_bullish


def get_market_pc_summary(df: pd.DataFrame) -> dict:
    """Aggregate P/C across all scanned tickers — market-wide sentiment."""
    if df.empty:
        return {}

    total_calls = df["call_vol"].sum()
    total_puts = df["put_vol"].sum()
    market_pc = round(total_puts / (total_calls + 1), 3)

    if market_pc >= 1.2:
        market_sentiment = "🐻 Market-wide fear / bearish positioning"
    elif market_pc <= 0.6:
        market_sentiment = "🐂 Market-wide complacency / bullish positioning"
    else:
        market_sentiment = "⚪ Balanced market sentiment"

    return {
        "market_pc_ratio": market_pc,
        "market_sentiment": market_sentiment,
        "total_call_vol": int(total_calls),
        "total_put_vol": int(total_puts),
        "tickers_scanned": len(df),
        "bearish_count": int((df["pc_vol_ratio"] >= 1.0).sum()),
        "bullish_count": int((df["pc_vol_ratio"] <= 0.7).sum()),
    }
