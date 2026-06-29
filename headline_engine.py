"""
Headline Engine — generates 10 AI-written market intelligence headlines
refreshed every hour during market hours (9 AM – 5 PM ET, weekdays).

Signal pipeline:
  1. Live yfinance: SPY / QQQ / IWM / VIX snapshot (always fresh)
  2. Session state: pre-loaded nightly cache data (richer context)
  3. Local file cache: background-worker pre-computed signals (fallback)

Claude Haiku converts the structured signals into newsroom-style sentences.
Cost: ~$0.001 per refresh × 8 market hours = ~$0.008 / trading day.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import cache_manager as cm

_ET = ZoneInfo("America/New_York")
_CACHE_KEY = "headline_stories"
_TTL = 3600          # 1 hour TTL
_MARKET_OPEN = 9     # 9 AM ET
_MARKET_CLOSE = 17   # 5 PM ET

_BULLET = "  ●  "   # ●


# ── Market-hours gate ─────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    now = datetime.now(_ET)
    return now.weekday() < 5 and _MARKET_OPEN <= now.hour < _MARKET_CLOSE


# ── Signal collectors ─────────────────────────────────────────────────────────

def _fetch_market_pulse() -> dict:
    """Live SPY / QQQ / IWM / VIX from yfinance — always fresh."""
    result: dict = {}
    try:
        data = yf.download(
            ["SPY", "QQQ", "IWM", "^VIX"],
            period="2d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        closes = data["Close"] if "Close" in data else data
        for sym in ["SPY", "QQQ", "IWM", "^VIX"]:
            col = sym
            if col not in closes.columns:
                continue
            c = closes[col].dropna()
            if len(c) >= 2:
                chg = (c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100
                key = sym.replace("^", "")
                result[key] = {
                    "price": round(float(c.iloc[-1]), 2),
                    "chg": round(float(chg), 2),
                }
    except Exception:
        pass
    return result


def _get_top_movers(ss: dict) -> list[dict]:
    """Top price movers: session state → local cache → empty."""
    # Try nightly price snapshots
    snaps = ss.get("nightly_price_snapshots") or {}
    if snaps:
        rows = [
            {"ticker": t, **v}
            for t, v in snaps.items()
            if v.get("ret_1w") is not None
        ]
        if rows:
            df = pd.DataFrame(rows)
            df["abs_ret"] = df["ret_1w"].abs()
            df = df.sort_values("abs_ret", ascending=False)
            return df.head(3).to_dict("records")

    # Fallback: local cache from background worker
    cached = cm.load("market_movers")
    if cached:
        return cached[:3]
    return []


def _get_unusual_flow(ss: dict) -> list[dict]:
    """Unusual options flow: session state → local cache → empty."""
    flow_df = ss.get("unusual_flow_df")
    if flow_df is not None and not flow_df.empty:
        cols = [c for c in ["ticker", "type", "strike", "expiration", "premium_$", "flow_type"] if c in flow_df.columns]
        return flow_df[cols].head(3).to_dict("records")

    stale = cm.load_stale("unusual_flow")
    if stale:
        data = stale[0] if isinstance(stale, tuple) else stale
        if isinstance(data, list):
            return data[:3]
    return []


def _get_sector_signals(ss: dict) -> dict:
    """Leading / lagging sectors: session state → local cache → empty."""
    sector_df = ss.get("sector_df")
    if sector_df is None or sector_df.empty:
        cached = cm.load("sector_rotation")
        sector_df = pd.DataFrame(cached) if cached else pd.DataFrame()

    if sector_df.empty or "sector" not in sector_df.columns or "ret_1m" not in sector_df.columns:
        return {}

    sector_df = sector_df.sort_values("ret_1m", ascending=False)
    return {
        "leader": sector_df.iloc[0]["sector"],
        "leader_ret": round(float(sector_df.iloc[0]["ret_1m"]), 1),
        "lagger": sector_df.iloc[-1]["sector"],
        "lagger_ret": round(float(sector_df.iloc[-1]["ret_1m"]), 1),
    }


def _get_upcoming_earnings(ss: dict) -> list[str]:
    """Tickers reporting earnings in the next 3 days."""
    cached = cm.load("earnings_calendar")
    if not cached:
        return []
    df = pd.DataFrame(cached)
    if df.empty or "date" not in df.columns:
        return []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.now()
    mask = df["date"].between(today, today + pd.Timedelta(days=3))
    col = "ticker" if "ticker" in df.columns else (df.columns[0] if not df.empty else None)
    if col:
        return df[mask][col].head(5).tolist()
    return []


def _get_congress_signal(ss: dict) -> dict:
    """Most recent congress trade: session state → local cache."""
    cdf = ss.get("congress_df")
    if cdf is None or cdf.empty:
        cached = cm.load("congress_trades")
        cdf = pd.DataFrame(cached) if cached else pd.DataFrame()

    if cdf.empty:
        stale = cm.load_stale("congress_trades")
        if stale:
            data = stale[0] if isinstance(stale, tuple) else stale
            cdf = pd.DataFrame(data) if data else pd.DataFrame()

    if cdf.empty or "politician" not in cdf.columns:
        return {}

    row = cdf.iloc[0]
    return {
        "politician": str(row.get("politician", "?")),
        "ticker": str(row.get("ticker", "?")),
        "transaction": str(row.get("transaction", "?")),
        "amount": str(row.get("amount", "?")),
    }


def _get_pc_signal(ss: dict) -> dict:
    """Most extreme put/call readings: session state → local cache."""
    pc_df = ss.get("pc_df")
    if pc_df is None or pc_df.empty:
        cached = cm.load("pc_ratios")
        if cached:
            rows = list(cached.values()) if isinstance(cached, dict) else cached
            pc_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if pc_df is None or pc_df.empty or "pc_vol_ratio" not in pc_df.columns:
        return {}

    pc_df["pc_vol_ratio"] = pd.to_numeric(pc_df["pc_vol_ratio"], errors="coerce")
    pc_df = pc_df.dropna(subset=["pc_vol_ratio"])
    if pc_df.empty:
        return {}

    bearish = pc_df.nlargest(1, "pc_vol_ratio").iloc[0]
    bullish = pc_df.nsmallest(1, "pc_vol_ratio").iloc[0]
    return {
        "bearish_ticker": str(bearish.get("ticker", "?")),
        "bearish_pc": round(float(bearish["pc_vol_ratio"]), 2),
        "bullish_ticker": str(bullish.get("ticker", "?")),
        "bullish_pc": round(float(bullish["pc_vol_ratio"]), 2),
    }


def _get_macro_signal(ss: dict) -> dict:
    """Current macro regime from session state."""
    ml = ss.get("macro_latest") or {}
    if not ml:
        return {}
    try:
        from macro_dashboard import classify_market_regime
        regime = classify_market_regime(ml)
        return {
            "regime": regime.get("regime", ""),
            "fed_rate": (ml.get("fed_funds") or {}).get("value", "?"),
            "cpi": (ml.get("cpi_yoy") or {}).get("value", "?"),
            "vix": (ml.get("vix") or {}).get("value", "?"),
        }
    except Exception:
        return {}


# ── Signal aggregator ─────────────────────────────────────────────────────────

def collect_signals(session_state: dict | None = None) -> dict:
    """
    Aggregates all available signals into one structured dict.
    Accepts optional Streamlit session_state for enrichment from nightly cache.
    """
    ss = session_state or {}
    return {
        "timestamp": datetime.now(_ET).strftime("%Y-%m-%d %H:%M ET"),
        "market_pulse": _fetch_market_pulse(),
        "top_movers": _get_top_movers(ss),
        "unusual_flow": _get_unusual_flow(ss),
        "sectors": _get_sector_signals(ss),
        "upcoming_earnings": _get_upcoming_earnings(ss),
        "congress": _get_congress_signal(ss),
        "pc_signal": _get_pc_signal(ss),
        "macro": _get_macro_signal(ss),
    }


# ── Headline generator ────────────────────────────────────────────────────────

def _build_data_block(signals: dict) -> str:
    pulse = signals.get("market_pulse", {})
    movers = signals.get("top_movers", [])
    flow = signals.get("unusual_flow", [])
    sectors = signals.get("sectors", {})
    earnings = signals.get("upcoming_earnings", [])
    congress = signals.get("congress", {})
    pc = signals.get("pc_signal", {})
    macro = signals.get("macro", {})

    def _p(sym):
        d = pulse.get(sym, {})
        return f"${d.get('price','?')} ({d.get('chg',0):+.2f}%)" if d else "N/A"

    mover_lines = "\n".join(
        f"  - {m.get('ticker','?')}: 1W {m.get('ret_1w', 0):+.1f}%, mkt cap ${m.get('market_cap',0)/1e9:.1f}B"
        for m in movers[:3]
    ) or "  - N/A"

    flow_lines = "\n".join(
        f"  - {f.get('ticker','?')} {str(f.get('type','?')).upper()} ${f.get('strike','?')} "
        f"exp {f.get('expiration','?')}: ${f.get('premium_$',0)/1e6:.1f}M, {f.get('flow_type','?')}"
        for f in flow[:3]
    ) or "  - None detected"

    return f"""
MARKET PULSE  ({signals['timestamp']}):
  SPY {_p('SPY')} | QQQ {_p('QQQ')} | IWM {_p('IWM')} | VIX {_p('VIX')}

TOP WEEKLY MOVERS:
{mover_lines}

UNUSUAL OPTIONS FLOW:
{flow_lines}

SECTOR ROTATION (1-month):
  Leading : {sectors.get('leader','?')} ({sectors.get('leader_ret',0):+.1f}%)
  Lagging : {sectors.get('lagger','?')} ({sectors.get('lagger_ret',0):+.1f}%)

UPCOMING EARNINGS (next 3 days):
  {', '.join(earnings) if earnings else 'None in range'}

CONGRESS TRADES (latest):
  {congress.get('politician','?')} — {congress.get('transaction','?')} {congress.get('ticker','?')} ({congress.get('amount','?')})

PUT/CALL EXTREMES:
  Most bearish: {pc.get('bearish_ticker','?')} P/C {pc.get('bearish_pc',0):.2f}
  Most bullish: {pc.get('bullish_ticker','?')} P/C {pc.get('bullish_pc',0):.2f}

MACRO REGIME:
  {macro.get('regime','?')} | Fed {macro.get('fed_rate','?')}% | CPI {macro.get('cpi','?')}% | VIX {macro.get('vix','?')}
"""


_SYSTEM = """\
You are a financial news wire editor at a professional market intelligence terminal.

Generate exactly 10 short, punchy headline sentences from the structured market data.

Rules:
- Each headline is ONE sentence, under 110 characters
- Be specific: cite exact tickers, percentages, and dollar figures from the data
- Vary tone and angle: mix price action, flow signals, sector trends, political trades
- Write in active present tense, professional register (Bloomberg / Reuters style)
- Headlines should feel like live news, not data summaries
- If data is missing or N/A for a field, skip that topic — do not fabricate
- Output ONLY the 10 headlines, one per line, no numbers, no bullets, no extra text"""


def generate_headlines(signals: dict) -> list[str]:
    """Call Claude Haiku to write 10 headlines. Falls back to rule-based if unavailable."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _rule_based_headlines(signals)

    data_block = _build_data_block(signals)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Generate 10 market intelligence headlines from this live data:\n{data_block}",
            }],
        )
        lines = [ln.strip() for ln in resp.content[0].text.strip().split("\n") if ln.strip()]
        if len(lines) >= 5:
            return lines[:10]
    except Exception:
        pass

    return _rule_based_headlines(signals)


def _rule_based_headlines(signals: dict) -> list[str]:
    """Deterministic fallback when the API is unavailable."""
    headlines: list[str] = []
    pulse = signals.get("market_pulse", {})
    sectors = signals.get("sectors", {})
    earnings = signals.get("upcoming_earnings", [])
    congress = signals.get("congress", {})
    movers = signals.get("top_movers", [])
    flow = signals.get("unusual_flow", [])
    pc = signals.get("pc_signal", {})
    macro = signals.get("macro", {})

    spy = pulse.get("SPY", {})
    if spy:
        direction = "advances" if spy.get("chg", 0) > 0 else "retreats"
        headlines.append(f"S&P 500 {direction} as SPY trades at ${spy['price']} ({spy['chg']:+.2f}% today)")

    vix = pulse.get("VIX", {})
    if vix:
        level = "elevated fear" if vix.get("price", 0) > 20 else "low volatility"
        headlines.append(f"VIX at {vix['price']} signals {level} — options market complacency watch")

    qqq = pulse.get("QQQ", {})
    iwm = pulse.get("IWM", {})
    if qqq and iwm:
        spread = round(qqq.get("chg", 0) - iwm.get("chg", 0), 2)
        if abs(spread) > 0.5:
            leader = "Tech (QQQ)" if spread > 0 else "Small-caps (IWM)"
            headlines.append(f"{leader} outperforming by {abs(spread):.2f}% today")

    if sectors.get("leader"):
        headlines.append(
            f"{sectors['leader']} leads sector rotation with {sectors['leader_ret']:+.1f}% 1-month return"
        )
    if sectors.get("lagger"):
        headlines.append(
            f"{sectors['lagger']} underperforms all sectors — down {abs(sectors['lagger_ret']):.1f}% over 30 days"
        )

    if earnings:
        headlines.append(f"Earnings watch: {', '.join(earnings[:3])} reporting within 72 hours")

    if congress.get("politician") and congress.get("ticker", "?") != "?":
        headlines.append(
            f"Congress: {congress['politician']} {congress['transaction']} {congress['ticker']} ({congress['amount']})"
        )

    if movers:
        m = movers[0]
        headlines.append(
            f"{m.get('ticker','?')} posts {m.get('ret_1w',0):+.1f}% 1-week move — on high-volume momentum"
        )

    if flow:
        f = flow[0]
        headlines.append(
            f"Smart money: ${f.get('premium_$',0)/1e6:.1f}M {f.get('type','?')} sweep detected in {f.get('ticker','?')}"
        )

    if pc.get("bearish_ticker"):
        headlines.append(
            f"Extreme bearish positioning: {pc['bearish_ticker']} put/call ratio hits {pc['bearish_pc']:.2f}"
        )

    if macro.get("regime"):
        headlines.append(f"Macro regime: {macro['regime']} — Fed at {macro.get('fed_rate','?')}%")

    while len(headlines) < 5:
        headlines.append("Market intelligence refreshing — check individual tabs for full analysis")

    return headlines[:10]


# ── Public API ────────────────────────────────────────────────────────────────

def get_cached_headlines(
    session_state: dict | None = None,
    force_refresh: bool = False,
) -> list[str]:
    """
    Returns up to 10 headlines from the 1-hour cache.
    Regenerates automatically when cache expires during market hours.
    Outside market hours returns last known headlines (stale-OK read).
    """
    if not force_refresh:
        fresh = cm.load(_CACHE_KEY)
        if fresh and isinstance(fresh, list) and len(fresh) >= 3:
            return fresh

    # Outside market hours: serve stale rather than regenerating
    if not is_market_hours() and not force_refresh:
        stale = cm.load_stale(_CACHE_KEY)
        if stale:
            data = stale[0] if isinstance(stale, tuple) else stale
            if isinstance(data, list) and data:
                return data
        return ["Market closed — pre-market intelligence will load at 9 AM ET"]

    signals = collect_signals(session_state)
    headlines = generate_headlines(signals)
    cm.save(_CACHE_KEY, headlines, ttl_seconds=_TTL)
    return headlines


def get_ticker_html(headlines: list[str], speed_seconds: int = 90) -> str:
    """
    Build a self-contained HTML string for a seamless CSS scrolling ticker.
    Content is duplicated so the loop is gapless.
    """
    items = f"{_BULLET}".join(headlines)
    # Duplicate for seamless loop: animation scrolls -50% which lands exactly at start
    double = f"{items}{_BULLET * 3}{items}{_BULLET * 3}"

    return f"""
<style>
  .mkt-ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 0;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }}
  .mkt-ticker-label {{
    display: inline-block;
    color: #f78166;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0 16px;
    vertical-align: middle;
  }}
  .mkt-ticker-scroll {{
    display: inline-block;
    white-space: nowrap;
    animation: mkt-scroll {speed_seconds}s linear infinite;
  }}
  .mkt-ticker-text {{
    color: #c9d1d9;
    font-size: 13.5px;
    letter-spacing: 0.2px;
  }}
  .mkt-ticker-text b {{
    color: #79c0ff;
  }}
  @keyframes mkt-scroll {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
  }}
</style>
<div class="mkt-ticker-wrap">
  <span class="mkt-ticker-label">&#x1F4E1; LIVE</span>
  <div class="mkt-ticker-scroll">
    <span class="mkt-ticker-text">{double}</span>
  </div>
</div>
"""
