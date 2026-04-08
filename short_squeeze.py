# short_squeeze.py
"""
Short Squeeze Scanner
Identifies large-cap stocks with elevated short interest, momentum, and
squeeze setup characteristics using free yfinance data.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value, default=None):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _fetch_ticker_data(ticker: str) -> dict | None:
    """Pull all squeeze-relevant metrics for one ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        market_cap = _safe(info.get("marketCap"), 0)
        price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"), 0)

        # Require large-cap (>$5B) and price > $10 to avoid penny / micro-cap noise
        if market_cap < 5_000_000_000 or price < 10:
            return None

        short_float_pct = _safe(info.get("shortPercentOfFloat"), 0)
        if short_float_pct is not None and short_float_pct > 1:
            # yfinance sometimes returns as decimal, sometimes as percent
            short_float_pct = short_float_pct / 100 if short_float_pct > 1 else short_float_pct
        short_float_pct = (short_float_pct or 0) * 100  # convert to readable %

        days_to_cover = _safe(info.get("shortRatio"), 0)
        shares_short = _safe(info.get("sharesShort"), 0)
        float_shares = _safe(info.get("floatShares"), 1) or 1

        # Price history for momentum / volume
        hist = yf.download(ticker, period="3mo", interval="1d",
                           progress=False, auto_adjust=True, threads=False)
        if hist is None or len(hist) < 22:
            return None

        # Flatten MultiIndex if needed
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()

        if len(close) < 22:
            return None

        current_price = float(close.iloc[-1])
        ma20 = float(close.tail(20).mean())
        ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None

        ret_3d = float((close.iloc[-1] / close.iloc[-4] - 1) * 100) if len(close) >= 4 else 0.0
        ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0
        ret_1mo = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0

        avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else 1.0
        avg_vol_5 = float(volume.tail(5).mean()) if len(volume) >= 5 else 1.0
        volume_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).tail(15)
        loss = (-delta.clip(upper=0)).tail(15)
        avg_gain = gain.mean()
        avg_loss = loss.mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50.0

        # Price vs MA
        pct_above_ma20 = (current_price / ma20 - 1) * 100

        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "Unknown")
        next_earnings = info.get("earningsDate") or info.get("earningsTimestamp")

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "price": round(current_price, 2),
            "market_cap_b": round(market_cap / 1e9, 1),
            "short_float_pct": round(short_float_pct, 1),
            "days_to_cover": round(days_to_cover or 0, 1),
            "shares_short_m": round((shares_short or 0) / 1e6, 1),
            "float_b": round(float_shares / 1e9, 2),
            "ret_3d": round(ret_3d, 1),
            "ret_5d": round(ret_5d, 1),
            "ret_1mo": round(ret_1mo, 1),
            "volume_ratio": round(volume_ratio, 2),
            "rsi": round(float(rsi), 1),
            "pct_above_ma20": round(pct_above_ma20, 1),
            "ma50_above": (ma50 is not None and current_price > ma50),
            "next_earnings": str(next_earnings) if next_earnings else None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_candidate(row: dict) -> tuple[float, list[str], str, str]:
    """
    Returns (score, reasons, setup_quality, buy_timing).
    Score is 0-100. Higher = better squeeze candidate.
    """
    score = 0.0
    reasons = []

    sf = row["short_float_pct"]
    dtc = row["days_to_cover"]
    vol_ratio = row["volume_ratio"]
    ret_3d = row["ret_3d"]
    ret_5d = row["ret_5d"]
    pct_ma20 = row["pct_above_ma20"]
    rsi = row["rsi"]

    # --- Short interest score (max 40 pts) ---
    if sf >= 30:
        score += 40
        reasons.append(f"Very high short float ({sf:.1f}%) — extremely crowded short position")
    elif sf >= 20:
        score += 30
        reasons.append(f"High short float ({sf:.1f}%) — significant short crowding")
    elif sf >= 15:
        score += 20
        reasons.append(f"Elevated short float ({sf:.1f}%) — notable short pressure")
    elif sf >= 10:
        score += 10
        reasons.append(f"Moderate short float ({sf:.1f}%)")

    # --- Days to cover (max 20 pts) ---
    if dtc >= 10:
        score += 20
        reasons.append(f"Extremely high days-to-cover ({dtc:.1f}d) — shorts trapped, slow exits")
    elif dtc >= 7:
        score += 15
        reasons.append(f"High days-to-cover ({dtc:.1f}d) — covering would take over a week")
    elif dtc >= 5:
        score += 10
        reasons.append(f"Solid days-to-cover ({dtc:.1f}d)")
    elif dtc >= 3:
        score += 5
        reasons.append(f"Moderate days-to-cover ({dtc:.1f}d)")

    # --- Momentum (max 25 pts) ---
    momentum_pts = 0
    if ret_3d > 5:
        momentum_pts += 12
        reasons.append(f"Strong 3-day momentum (+{ret_3d:.1f}%) — price forcing short covers")
    elif ret_3d > 2:
        momentum_pts += 7
        reasons.append(f"Positive 3-day momentum (+{ret_3d:.1f}%)")
    elif ret_3d > 0:
        momentum_pts += 3

    if ret_5d > 8:
        momentum_pts += 8
        reasons.append(f"Strong 5-day trend (+{ret_5d:.1f}%)")
    elif ret_5d > 3:
        momentum_pts += 4

    if pct_ma20 > 5:
        momentum_pts += 5
        reasons.append(f"Well above 20-day MA (+{pct_ma20:.1f}%) — uptrend confirmed")
    elif pct_ma20 > 0:
        momentum_pts += 2
        reasons.append(f"Above 20-day MA (+{pct_ma20:.1f}%) — bullish bias")
    elif pct_ma20 > -3:
        momentum_pts += 1  # near MA, potential breakout

    score += min(momentum_pts, 25)

    # --- Volume (max 15 pts) ---
    if vol_ratio >= 2.0:
        score += 15
        reasons.append(f"Volume surge ({vol_ratio:.1f}x avg) — institutional buying forcing covers")
    elif vol_ratio >= 1.5:
        score += 10
        reasons.append(f"Above-average volume ({vol_ratio:.1f}x avg) — rising interest")
    elif vol_ratio >= 1.2:
        score += 5
        reasons.append(f"Slightly elevated volume ({vol_ratio:.1f}x avg)")

    # --- RSI context (no extra pts, used for timing) ---
    if 40 <= rsi <= 65:
        reasons.append(f"RSI {rsi:.0f} — not overbought, squeeze has room to run")
    elif rsi > 70:
        reasons.append(f"RSI {rsi:.0f} — overbought caution; don't chase, wait for pullback")
    elif rsi < 35:
        reasons.append(f"RSI {rsi:.0f} — oversold despite short pressure; potential coiled spring")

    # --- Setup quality ---
    if score >= 75:
        setup_quality = "Tier 1 — Prime Squeeze Setup"
    elif score >= 55:
        setup_quality = "Tier 2 — Strong Candidate"
    elif score >= 35:
        setup_quality = "Tier 3 — Watch List"
    else:
        setup_quality = "Low Priority"

    # --- Buy timing ---
    buy_timing = _derive_buy_timing(row, rsi, score)

    return round(score, 1), reasons, setup_quality, buy_timing


def _derive_buy_timing(row: dict, rsi: float, score: float) -> str:
    pct_ma20 = row["pct_above_ma20"]
    vol_ratio = row["volume_ratio"]
    ret_3d = row["ret_3d"]
    dtc = row["days_to_cover"]

    # Overbought + already surged — wait
    if rsi > 72 and ret_3d > 10:
        return (
            "WAIT — Stock already surged >10% in 3 days with RSI >72. "
            "Chasing here risks buying the peak. Watch for a pullback to the 20-day MA "
            "or a 2-3 day consolidation with volume drop, then re-enter on renewed volume."
        )

    # Below MA, no momentum yet — catalyst needed
    if pct_ma20 < -5 and vol_ratio < 1.2:
        return (
            "WATCH — Below 20-day MA with no volume confirmation. "
            "High short interest is a necessary but not sufficient condition. "
            f"Wait for price to reclaim the 20-day MA on volume (≥1.5x avg) — "
            "that is your trigger. Set an alert and don't enter early."
        )

    # Near MA, potential breakout about to trigger
    if -3 <= pct_ma20 <= 3 and vol_ratio >= 1.3:
        return (
            "BUY ZONE (Breakout Setup) — Price is right at the 20-day MA with rising volume. "
            "Enter on a clean break and daily close above the MA. "
            "Ideal window: 10:00–11:30 AM ET on a day the broader market is green. "
            "Avoid entering in the first 10 minutes of market open."
        )

    # Above MA, momentum building — buy on pullbacks
    if pct_ma20 > 3 and vol_ratio >= 1.5 and rsi < 68:
        timing = "BUY ZONE (Momentum Confirmed)"
        if dtc >= 7:
            timing += (
                f" — Uptrend confirmed above 20-day MA with high days-to-cover ({dtc:.0f}d). "
                "Shorts are trapped and exits take time. Best entry: "
                "any intraday dip to the 20-day MA or prior resistance (now support). "
                "Optimal time: 10:00–11:30 AM ET after the opening volatility settles."
            )
        else:
            timing += (
                " — Above MA with volume confirmation. "
                "Enter on shallow pullbacks during the session (not gap-up opens). "
                "Best window: 10:00–11:30 AM ET."
            )
        return timing

    # Above MA, low volume — wait for volume confirmation
    if pct_ma20 > 3 and vol_ratio < 1.3:
        return (
            "WAIT (Volume Missing) — Price trend is positive and above 20-day MA, "
            "but volume has not confirmed the move. Without volume, shorts feel no urgency to cover. "
            "Watch for a day where volume spikes ≥1.5x the 20-day average — "
            "that is your entry signal. Set a volume alert."
        )

    # Default — moderate setup
    return (
        "MONITOR — Setup shows promise but needs one more confirmation. "
        "Watch for: (1) price clearing a recent resistance level, "
        "(2) volume expanding ≥1.5x average, or (3) a news catalyst. "
        "Best entry window when triggered: 10:00–11:30 AM ET."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_short_squeeze(tickers: list[str], min_short_float: float = 8.0,
                       workers: int = 12) -> pd.DataFrame:
    """
    Scan a list of tickers and return a ranked short squeeze candidate table.

    Parameters
    ----------
    tickers : list of ticker symbols
    min_short_float : minimum short float % to include (default 8%)
    workers : thread pool size

    Returns
    -------
    pd.DataFrame sorted by squeeze_score descending
    """
    rows = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_ticker_data, t): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result is not None and result["short_float_pct"] >= min_short_float:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        score, reasons, setup_quality, buy_timing = _score_candidate(row)
        records.append({
            **row,
            "squeeze_score": score,
            "setup_quality": setup_quality,
            "reasons": " | ".join(reasons),
            "buy_timing": buy_timing,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("squeeze_score", ascending=False).reset_index(drop=True)
    return df
