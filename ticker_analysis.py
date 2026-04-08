# ticker_analysis.py
"""
Single-Ticker Deep Analysis
Fetches fresh data from Polygon + Finviz and produces a complete
analysis report for one ticker: technicals, fundamentals, options,
squeeze setup, intraday timing, strategy recommendation, and summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import polygon_client as _polygon
from build_features import build_features

try:
    from finvizfinance.quote import finvizfinance as _finviz_quote
    _FINVIZ_AVAILABLE = True
except Exception:
    _FINVIZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=None):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _pct(val):
    return f"{val:+.2f}%" if val is not None else "—"


def _fmt(val, decimals=2):
    return f"{val:.{decimals}f}" if val is not None else "—"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _get_price_history(ticker: str) -> pd.DataFrame:
    """1-year daily OHLCV. Polygon first, yfinance fallback."""
    if _polygon.available():
        df = _polygon.get_daily_bars(ticker, days=365)
        if df is not None and len(df) >= 50:
            return df
    tk = yf.Ticker(ticker)
    df = tk.history(period="1y", interval="1d", auto_adjust=True)
    return df if df is not None else pd.DataFrame()


def _get_intraday(ticker: str) -> pd.DataFrame:
    """60-day 5-min bars. Polygon first, yfinance fallback."""
    if _polygon.available():
        df = _polygon.get_intraday_bars(ticker, days=60)
        if df is not None and len(df) >= 50:
            return df
    import yfinance as yf
    raw = yf.download(ticker, period="60d", interval="5m",
                      progress=False, auto_adjust=True, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("America/New_York")
    else:
        raw.index = raw.index.tz_localize("UTC").tz_convert("America/New_York")
    return raw


def _get_finviz(ticker: str) -> dict:
    if not _FINVIZ_AVAILABLE:
        return {}
    import time
    for attempt in range(3):
        try:
            return _finviz_quote(ticker).ticker_fundament()
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "too many" in msg or "429" in msg:
                time.sleep(2 + attempt * 2)  # 2s, 4s, 6s
                continue
            return {}  # non-rate-limit error, give up immediately
    return {}  # all retries exhausted — return empty, don't block analysis


def _get_snapshot(ticker: str) -> dict:
    if _polygon.available():
        snap = _polygon.get_snapshot(ticker)
        if snap:
            return snap
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    return {
        "price": _safe(info.get("currentPrice") or info.get("regularMarketPrice")),
        "change_pct": _safe(info.get("regularMarketChangePercent")),
        "volume": _safe(info.get("regularMarketVolume")),
        "prev_close": _safe(info.get("previousClose")),
    }


# ---------------------------------------------------------------------------
# Technical analysis
# ---------------------------------------------------------------------------

def _technicals(history: pd.DataFrame) -> dict:
    if history.empty:
        return {}

    feats = build_features(history)
    if feats.empty:
        return {}

    latest = feats.iloc[-1]

    close = history["Close"].dropna()
    if isinstance(close.iloc[0], pd.Series):
        close = close.iloc[:, 0]
    close = close.astype(float)

    price = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    pct_from_high = (price / high_52w - 1) * 100
    pct_from_low = (price / low_52w - 1) * 100

    # Trend
    trend = "Uptrend"
    if ma50 and ma200:
        if price > ma50 > ma200:
            trend = "Strong Uptrend"
        elif price < ma50 < ma200:
            trend = "Strong Downtrend"
        elif price < ma50:
            trend = "Downtrend"
    elif ma50:
        trend = "Uptrend" if price > ma50 else "Downtrend"

    rsi = _safe(latest.get("rsi_14"), 50.0)
    adx = _safe(latest.get("adx_14"), 20.0)
    macd_hist = _safe(latest.get("macd_hist"), 0.0)
    atr = _safe(latest.get("atr_14"), price * 0.02)
    vol_ratio = _safe(latest.get("volume_ratio_5"), 1.0)
    ret_5d = _safe(latest.get("close_ret_5d"), 0.0)
    ret_1mo = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0

    # Support / resistance: recent swing lows/highs
    recent = close.tail(30)
    support = float(recent.min())
    resistance = float(recent.max())

    # RSI signal
    if rsi >= 70:
        rsi_signal = "Overbought — consider waiting for pullback"
    elif rsi <= 30:
        rsi_signal = "Oversold — potential bounce setup"
    elif 50 <= rsi < 70:
        rsi_signal = "Bullish momentum"
    else:
        rsi_signal = "Bearish momentum"

    # MACD signal
    macd_signal = "Bullish (histogram positive)" if macd_hist > 0 else "Bearish (histogram negative)"

    # ADX
    adx_signal = (
        "Strong trend (ADX > 25)" if adx >= 25 else
        "Weak / no trend (ADX < 20)" if adx < 20 else
        "Developing trend"
    )

    return {
        "price": price,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low,
        "trend": trend,
        "rsi": rsi,
        "rsi_signal": rsi_signal,
        "adx": adx,
        "adx_signal": adx_signal,
        "macd_hist": macd_hist,
        "macd_signal": macd_signal,
        "atr": atr,
        "vol_ratio_5d": vol_ratio,
        "ret_5d": ret_5d,
        "ret_1mo": ret_1mo,
        "support": support,
        "resistance": resistance,
    }


# ---------------------------------------------------------------------------
# Fundamentals from Finviz
# ---------------------------------------------------------------------------

def _fundamentals(fv: dict) -> dict:
    def _fpct(key):
        raw = fv.get(key, "").replace("%", "").strip()
        return _safe(raw)

    def _fnum(key):
        raw = fv.get(key, "").replace(",", "").strip()
        return _safe(raw)

    price = _fnum("Price")
    target = _fnum("Target Price")
    upside = round((target / price - 1) * 100, 1) if target and price else None

    recom = _fnum("Recom")
    recom_label = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
    recom_str = recom_label.get(round(recom) if recom else 0, f"{recom:.2f}" if recom else "—")

    return {
        "company": fv.get("Company", ""),
        "sector": fv.get("Sector", ""),
        "industry": fv.get("Industry", ""),
        "market_cap": fv.get("Market Cap", "—"),
        "pe": _fnum("P/E"),
        "forward_pe": _fnum("Forward P/E"),
        "eps_ttm": _fnum("EPS (ttm)"),
        "eps_next_y": _fpct("EPS next Y Percentage"),
        "eps_growth_5y": _fpct("EPS next 5Y"),
        "revenue_growth": _fpct("Sales Y/Y TTM"),
        "profit_margin": _fpct("Profit Margin"),
        "roe": _fpct("ROE"),
        "debt_eq": _fnum("Debt/Eq"),
        "beta": _fnum("Beta"),
        "short_float_pct": _fpct("Short Float"),
        "short_ratio": _fnum("Short Ratio"),
        "insider_trans": _fpct("Insider Trans"),
        "inst_own": _fpct("Inst Own"),
        "target_price": target,
        "target_upside_pct": upside,
        "analyst_recom": recom,
        "analyst_recom_str": recom_str,
        "earnings_date": fv.get("Earnings", "—"),
        "dividend": fv.get("Dividend Est.", "—"),
        "52w_high": fv.get("52W High", "—"),
        "52w_low": fv.get("52W Low", "—"),
    }


# ---------------------------------------------------------------------------
# Intraday timing summary
# ---------------------------------------------------------------------------

def _timing_summary(ticker: str, direction: str) -> dict:
    """Run the timing engine and return a condensed result."""
    try:
        from timing_engine import get_intraday_timing
        return get_intraday_timing(ticker, direction)
    except Exception as e:
        return {"error": str(e), "best_entry_window": "—", "best_exit_window": "—",
                "timing_note": "Timing unavailable.", "gap_fade_prob": None,
                "volume_profile": [], "avg_return_by_slot": [],
                "win_rate_by_slot": [], "slot_labels": [], "n_days": 0}


# ---------------------------------------------------------------------------
# Overall signal and trade idea
# ---------------------------------------------------------------------------

def _build_signal(tech: dict, fund: dict, snap: dict) -> dict:
    """
    Combine all signals into a directional bias, confidence, and trade idea.
    """
    score = 0
    reasons_bull = []
    reasons_bear = []

    rsi = tech.get("rsi", 50)
    adx = tech.get("adx", 20)
    macd = tech.get("macd_hist", 0)
    ret_5d = tech.get("ret_5d", 0)
    vol_ratio = tech.get("vol_ratio_5d", 1)
    trend = tech.get("trend", "")
    price = tech.get("price", 0)
    ma20 = tech.get("ma20", price)
    ma50 = tech.get("ma50", price)
    pct_from_high = tech.get("pct_from_high", -50)
    target_upside = fund.get("target_upside_pct")
    analyst = fund.get("analyst_recom")
    short_float = fund.get("short_float_pct")
    change_pct = snap.get("change_pct", 0) or 0

    # Trend
    if "Strong Uptrend" in trend:
        score += 3; reasons_bull.append("Strong uptrend (price > MA50 > MA200)")
    elif "Uptrend" in trend:
        score += 1; reasons_bull.append("Uptrend (price above key MAs)")
    elif "Strong Downtrend" in trend:
        score -= 3; reasons_bear.append("Strong downtrend (price < MA50 < MA200)")
    elif "Downtrend" in trend:
        score -= 1; reasons_bear.append("Downtrend (price below key MAs)")

    # RSI
    if 50 <= rsi <= 68:
        score += 1; reasons_bull.append(f"RSI {rsi:.0f} — bullish momentum, not overbought")
    elif rsi > 70:
        score -= 1; reasons_bear.append(f"RSI {rsi:.0f} — overbought, risk of pullback")
    elif rsi < 35:
        score += 1; reasons_bull.append(f"RSI {rsi:.0f} — oversold, potential bounce")
    elif rsi < 50:
        score -= 1; reasons_bear.append(f"RSI {rsi:.0f} — below midline, bearish bias")

    # MACD
    if macd > 0:
        score += 1; reasons_bull.append("MACD histogram positive — bullish momentum")
    else:
        score -= 1; reasons_bear.append("MACD histogram negative — bearish momentum")

    # ADX
    if adx >= 25:
        score += 1; reasons_bull.append(f"ADX {adx:.0f} — strong trend in place")

    # Volume
    if vol_ratio >= 1.5:
        score += 1; reasons_bull.append(f"Volume {vol_ratio:.1f}x average — institutional activity")
    elif vol_ratio < 0.7:
        score -= 1; reasons_bear.append(f"Volume {vol_ratio:.1f}x average — low conviction")

    # 5d return
    if ret_5d > 3:
        score += 1; reasons_bull.append(f"Strong 5-day return (+{ret_5d:.1f}%)")
    elif ret_5d < -3:
        score -= 1; reasons_bear.append(f"Weak 5-day return ({ret_5d:.1f}%)")

    # 52w proximity
    if pct_from_high >= -5:
        score += 1; reasons_bull.append("Near 52-week high — breakout potential")
    elif pct_from_high < -30:
        score -= 1; reasons_bear.append(f"Far from 52-week high ({pct_from_high:.0f}%) — laggard")

    # Analyst
    if analyst and analyst <= 2.0:
        score += 2; reasons_bull.append(f"Analyst consensus: {fund.get('analyst_recom_str')} ({analyst:.1f})")
    elif analyst and analyst >= 4.0:
        score -= 2; reasons_bear.append(f"Analyst consensus: {fund.get('analyst_recom_str')} ({analyst:.1f})")

    # Target price
    if target_upside and target_upside > 15:
        score += 1; reasons_bull.append(f"Analyst target implies +{target_upside:.0f}% upside")
    elif target_upside and target_upside < -5:
        score -= 1; reasons_bear.append(f"Analyst target implies {target_upside:.0f}% downside")

    # Short float (squeeze potential)
    if short_float and short_float >= 10:
        score += 1; reasons_bull.append(f"High short interest ({short_float:.1f}%) — squeeze potential")

    # Today's move
    if change_pct > 2:
        score += 1; reasons_bull.append(f"Strong today +{change_pct:.1f}% — momentum day")
    elif change_pct < -2:
        score -= 1; reasons_bear.append(f"Weak today {change_pct:.1f}% — selling pressure")

    # Direction
    if score >= 4:
        direction = "Strong Bullish"
        contract = "call"
        confidence = min(95, 60 + score * 5)
    elif score >= 2:
        direction = "Bullish"
        contract = "call"
        confidence = min(80, 55 + score * 5)
    elif score <= -4:
        direction = "Strong Bearish"
        contract = "put"
        confidence = min(95, 60 + abs(score) * 5)
    elif score <= -2:
        direction = "Bearish"
        contract = "put"
        confidence = min(80, 55 + abs(score) * 5)
    else:
        direction = "Neutral / Mixed"
        contract = "hold"
        confidence = 40

    # Trade idea
    atr = tech.get("atr", price * 0.02)
    if contract != "hold" and price > 0 and atr > 0:
        if contract == "call":
            entry = f"Buy call near ${price:.2f}. Entry on pullback to ${price - atr * 0.5:.2f} or morning breakout above ${price + atr * 0.3:.2f}."
            stop = f"Exit if {ticker_placeholder} drops below ${price - atr * 1.5:.2f} (1.5× ATR stop)."
            target_str = f"Target ${price + atr * 2:.2f} to ${price + atr * 3:.2f} (2–3× ATR)."
        else:
            entry = f"Buy put near ${price:.2f}. Enter on bounce to ${price + atr * 0.5:.2f} or break below ${price - atr * 0.3:.2f}."
            stop = f"Exit if {ticker_placeholder} rallies above ${price + atr * 1.5:.2f} (1.5× ATR stop)."
            target_str = f"Target ${price - atr * 2:.2f} to ${price - atr * 3:.2f} (2–3× ATR)."
        trade_idea = f"{entry} {stop} {target_str}"
    else:
        trade_idea = "No clear directional setup. Wait for trend confirmation before entering."

    return {
        "score": score,
        "direction": direction,
        "contract": contract,
        "confidence": confidence,
        "bullish_reasons": reasons_bull,
        "bearish_reasons": reasons_bear,
        "trade_idea": trade_idea,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ticker_placeholder = "ticker"  # replaced at call time


def analyse_ticker(ticker: str) -> dict:
    """
    Run a full analysis on a single ticker.
    Returns a structured dict with all sections.
    """
    global ticker_placeholder
    ticker_placeholder = ticker

    ticker = ticker.upper().strip()

    snap = _get_snapshot(ticker)
    history = _get_price_history(ticker)

    try:
        fv_raw = _get_finviz(ticker)
    except Exception:
        fv_raw = {}

    tech = _technicals(history)
    fund = _fundamentals(fv_raw)

    direction = "call" if tech.get("rsi", 50) >= 50 else "put"
    timing = _timing_summary(ticker, direction)

    signal = _build_signal(tech, fund, snap)

    # Plain-English summary
    price = snap.get("price") or tech.get("price", 0)
    change = snap.get("change_pct") or 0
    summary_lines = [
        f"**{ticker}** — {fund.get('company', ticker)} ({fund.get('sector', '')})",
        f"Current price: **${price:.2f}** ({change:+.2f}% today)",
        f"Overall signal: **{signal['direction']}** | Confidence: **{signal['confidence']}%**",
        "",
        f"Trend: {tech.get('trend', '—')} | RSI: {tech.get('rsi', '—'):.0f} | ADX: {tech.get('adx', '—'):.0f}",
        f"Analyst: {fund.get('analyst_recom_str', '—')} | Target: ${fund.get('target_price') or '—'} ({_pct(fund.get('target_upside_pct'))} upside)",
        f"Earnings: {fund.get('earnings_date', '—')} | Short Float: {fund.get('short_float_pct') or '—'}%",
        "",
        f"Best entry window: **{timing.get('best_entry_window', '—')}**",
        f"Best exit window: **{timing.get('best_exit_window', '—')}**",
        "",
        "**Trade Idea:**",
        signal["trade_idea"],
    ]

    return {
        "ticker": ticker,
        "snapshot": snap,
        "technicals": tech,
        "fundamentals": fund,
        "signal": signal,
        "timing": timing,
        "summary": "\n".join(summary_lines),
        "history": history,
    }
