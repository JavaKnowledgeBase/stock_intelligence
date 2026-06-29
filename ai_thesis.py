"""
AI Trade Thesis Generator — synthesises all available platform data for a
ticker into a structured, actionable trade thesis via Claude.

Uses claude-haiku-4-5 by default (fast, cheap). Caller can pass sonnet for
deeper analysis. Requires ANTHROPIC_API_KEY in environment.
"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf


# ── Context builders ──────────────────────────────────────────────────────────

def _price_context(ticker: str) -> str:
    """Fetch live price, fundamentals, and recent returns from yfinance."""
    try:
        t = yf.Ticker(ticker)
        spot = getattr(t.fast_info, "last_price", None) or 0.0
        info = t.info or {}

        hist = t.history(period="3mo", auto_adjust=True)
        ret_1w = ret_1m = ret_3m = vol_ann = None
        if not hist.empty:
            c = hist["Close"]
            if len(c) >= 5:
                ret_1w = (c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] * 100
            if len(c) >= 21:
                ret_1m = (c.iloc[-1] - c.iloc[-21]) / c.iloc[-21] * 100
            ret_3m = (c.iloc[-1] - c.iloc[0]) / c.iloc[0] * 100
            rets = c.pct_change().dropna()
            vol_ann = rets.std() * (252 ** 0.5) * 100
            avg_vol = hist["Volume"].mean()
            recent_vol = hist["Volume"].iloc[-5:].mean()
            vol_trend = "above average" if recent_vol > avg_vol * 1.1 else \
                        "below average" if recent_vol < avg_vol * 0.9 else "average"
        else:
            vol_trend = "unknown"

        lines = [
            f"## {ticker} — Price & Fundamentals",
            f"- Spot: ${spot:,.2f}",
            f"- Sector: {info.get('sector', '?')} / {info.get('industry', '?')}",
            f"- Market Cap: ${info.get('marketCap', 0)/1e9:.1f}B",
            f"- Trailing P/E: {info.get('trailingPE', 'N/A')} | Forward P/E: {info.get('forwardPE', 'N/A')}",
            f"- 52W Range: ${info.get('fiftyTwoWeekLow', 0):.2f} – ${info.get('fiftyTwoWeekHigh', 0):.2f}",
            f"- Beta: {info.get('beta', 'N/A')}",
            f"- Analyst Target: ${info.get('targetMeanPrice', 0):.2f} ({info.get('recommendationKey', 'N/A')})",
        ]
        if ret_1w is not None:
            lines += [
                f"- Returns: 1W {ret_1w:+.1f}% | 1M {ret_1m:+.1f}% | 3M {ret_3m:+.1f}%",
                f"- Annualised Vol: {vol_ann:.1f}% | Recent volume: {vol_trend}",
            ]
        return "\n".join(lines)
    except Exception as e:
        return f"## {ticker} — Price context unavailable ({e})"


def _session_context(ticker: str, ss: dict) -> str:
    """Pull relevant data already loaded in the Streamlit session."""
    parts = []
    t = ticker.upper()

    # Earnings analytics
    ea = ss.get("ea_result") or {}
    if ea.get("ticker", "").upper() == t and not ea.get("error"):
        straddle = ea.get("straddle") or {}
        parts.append(
            f"## Earnings Analytics\n"
            f"- Next earnings: {ea.get('next_earnings_date', '?')} "
            f"({ea.get('days_to_earnings', '?')} days)\n"
            f"- Implied move: ±{straddle.get('implied_move_pct', 0):.1f}% "
            f"(straddle ${straddle.get('straddle_cost', 0):.2f})\n"
            f"- Avg historical post-earnings move: ±{ea.get('avg_hist_move', 0):.1f}%\n"
            f"- EPS beat rate: {ea.get('beat_rate', 0):.0f}% "
            f"({ea.get('beat_count', 0)}/{ea.get('total_quarters', 0)} qtrs)\n"
            f"- IV Rank: {ea.get('iv_rank', 'N/A')}/100\n"
            f"- Upside BE: ${straddle.get('upside_be', 0):.2f} | "
            f"Downside BE: ${straddle.get('downside_be', 0):.2f}"
        )

    # GEX
    gex = ss.get("gex_data") or {}
    if ss.get("gex_ticker", "").upper() == t and gex:
        net = gex.get("net_gex", 0)
        parts.append(
            f"## Dealer Gamma Exposure (GEX)\n"
            f"- Net GEX: ${net/1e6:.1f}M ({'positive / stabilising' if net >= 0 else 'negative / amplifying'})\n"
            f"- Call Wall (resistance): ${gex.get('call_wall', 0):.0f}\n"
            f"- Put Wall (support): ${gex.get('put_wall', 0):.0f}\n"
            f"- Gamma Flip: ${gex.get('gamma_flip') or 0:.0f}\n"
            f"- Dealer Bias: {gex.get('dealer_bias', '?')}"
        )

    # Unusual flow
    flow_df = ss.get("unusual_flow_df")
    if flow_df is not None and not flow_df.empty and "ticker" in flow_df.columns:
        tf = flow_df[flow_df["ticker"] == t]
        if not tf.empty:
            total_prem = tf["premium_$"].sum()
            top = tf.iloc[0]
            parts.append(
                f"## Unusual Options Flow\n"
                f"- Total premium detected: ${total_prem:,.0f}\n"
                f"- Signals: {len(tf)} ({(tf['type']=='call').sum()} calls, "
                f"{(tf['type']=='put').sum()} puts)\n"
                f"- Largest: {top.get('flow_type','')} {top.get('type','')} "
                f"${top.get('strike',0)} exp {top.get('expiration','')} "
                f"Vol/OI {top.get('vol_oi_ratio',0):.1f}x"
            )

    # Congress trades
    cdf = ss.get("congress_df")
    if cdf is not None and not cdf.empty and "ticker" in cdf.columns:
        ct = cdf[cdf["ticker"] == t]
        if not ct.empty:
            buys = (ct["transaction"] == "🟢 Buy").sum()
            sells = (ct["transaction"] == "🔴 Sell").sum()
            pols = ", ".join(ct["politician"].unique()[:3])
            parts.append(
                f"## Congressional Trades\n"
                f"- Buys: {buys} | Sells: {sells}\n"
                f"- Politicians: {pols}"
            )

    # Macro regime
    ml = ss.get("macro_latest") or {}
    if ml:
        try:
            from macro_dashboard import classify_market_regime
            regime = classify_market_regime(ml)
            fed_v = (ml.get("fed_funds") or {}).get("value", "?")
            cpi_v = (ml.get("cpi_yoy") or {}).get("value", "?")
            vix_v = (ml.get("vix") or {}).get("value", "?")
            spread_v = (ml.get("yield_spread") or {}).get("value", "?")
            parts.append(
                f"## Macro Context\n"
                f"- Regime: {regime['regime']}\n"
                f"- Fed Funds: {fed_v}% | CPI YoY: {cpi_v}%\n"
                f"- 10Y-2Y Spread: {spread_v}% | VIX: {vix_v}"
            )
        except Exception:
            pass

    # Hedge fund 13F (single fund loaded)
    hf_df = ss.get("hf_holdings_df")
    hf_name = ss.get("hf_fund_name", "")
    if hf_df is not None and not hf_df.empty and "company" in hf_df.columns:
        match = hf_df[hf_df["company"].str.upper().str.contains(t, na=False)]
        if not match.empty:
            row = match.iloc[0]
            parts.append(
                f"## Hedge Fund 13F ({hf_name})\n"
                f"- Holding: {row['company']} — ${row['value_usd']/1e6:.1f}M "
                f"({row['pct_portfolio']:.1f}% of portfolio)"
            )

    return "\n\n".join(parts) if parts else ""


_SYSTEM = """\
You are a senior portfolio manager at a multi-strategy hedge fund generating trade briefs for the trading desk.

Rules:
- Be precise: cite exact numbers from the data provided
- Be balanced: one bear risk for every bull signal
- Never mention the platform or "the data provided" — speak as if you computed it yourself
- Use concrete price levels, not vague ranges
- Format using the exact template below — no extra sections

TEMPLATE:
## Setup
[One sentence: bull/bear/neutral, primary signal, time horizon]

## Bull Case
- [data-backed point]
- [data-backed point]

## Bear Case / Risks
- [data-backed point]
- [data-backed point]

## Trade Structure
| Field | Value |
|-------|-------|
| Direction | Bull / Bear / Neutral |
| Time Horizon | X days / weeks |
| Entry | $X |
| Target | $X (+X%) |
| Stop | $X (-X%) |
| R/R | X:1 |
| Sizing | Small / Medium / Full (justify briefly) |

## Options Play
[Specific strategy: e.g. "Buy $X call expiring YYYY-MM-DD, cost $X, breakeven $X" — or "N/A" if no options data]

## One Sentence Summary
[The single most important thing a trader needs to know right now]
"""


def generate_thesis(
    ticker: str,
    session_state: dict | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """
    Generate a structured AI trade thesis for ticker.
    Returns markdown string (thesis or error message).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "_ANTHROPIC_API_KEY not set. Add it to Streamlit Cloud secrets._"

    price_ctx = _price_context(ticker)
    session_ctx = _session_context(ticker, session_state or {})

    user_msg = (
        f"Generate a trade thesis for **{ticker}**.\n\n"
        f"{price_ctx}\n\n"
        f"{session_ctx}\n\n"
        f"Today: {datetime.now().strftime('%Y-%m-%d')}"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"_Error: {e}_"
