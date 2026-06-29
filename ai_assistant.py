"""
AI Assistant — context-aware chatbot for the Stock Intelligence Platform.

Powered by Anthropic Claude. Knows:
  - Every feature and tab in the app
  - Live analysis data currently loaded (squeeze scan, options flow, strategy, etc.)
  - User's watchlist / portfolio (pasted by the user)
  - Full conversation history (multi-turn)

Set ANTHROPIC_API_KEY in .env to enable.
"""
from __future__ import annotations

import os
from typing import Generator, Optional

import pandas as pd

import ssl_fix  # patches CURL_CA_BUNDLE + REQUESTS_CA_BUNDLE + truststore

_AVAILABLE = False
_client = None

try:
    import anthropic as _anthropic
    _AVAILABLE = bool(os.getenv("ANTHROPIC_API_KEY"))
    if _AVAILABLE:
        _client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except ImportError:
    pass

# ── Model selection ────────────────────────────────────────────────────────────
MODELS = {
    "Fast (Haiku)":   "claude-haiku-4-5-20251001",
    "Smart (Sonnet)": "claude-sonnet-4-6",
}
DEFAULT_MODEL = "Fast (Haiku)"

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are the AI assistant for the **Institutional Options Intelligence Platform** — a professional stock research and trading analysis tool.

## What this platform does
The app has 12 analysis tabs:
1. **Market Options Screener** — scans 60+ tickers for liquid options contracts, filterable by call/put, ITM, volume.
2. **Market Volume Leaders** — ranks tickers by options volume, shows call/put dominance and weekly trend.
3. **Rapid Movers** — detects tickers with unusual 1W/1M momentum using 60+ technical indicators.
4. **Strategy Ideas** — AI-generated options strategies (calls/puts, strike, expiry) with entry/stop/take-profit rules, confidence scores.
5. **Options Chain Explorer** — full options chain for any ticker + unusual flow detection + Dealer Gamma Exposure (GEX).
6. **Price Forecast** — LightGBM ML models (trained on 27 tickers) predicting direction bias with confidence %.
7. **Short Squeeze Scanner** — scores tickers 0–100 on squeeze potential: short float %, days-to-cover, momentum, volume ratio. Tier 1/2/3 quality.
8. **Intraday Timing** — optimal 30-min entry/exit windows from 60 days of 5-min data, gap fade probability.
9. **Ticker Analysis** — full deep dive: snapshot, technicals, fundamentals (P/E, EPS growth, analyst rating), timing.
10. **Earnings Calendar** — upcoming earnings with IV/HV ratio. IV/HV ≥ 1.25 = elevated premium (earnings uncertainty priced in).
11. **Market Screener** — multi-factor technical screener with 5 presets: Oversold Bounce, Breakout Setup, High Momentum, Short Squeeze Risk, Pullback in Uptrend.
12. **Insider Flow** — recent Form 4 insider buy/sell transactions from SEC EDGAR.

## Key concepts you explain fluently
- **RSI** (0–100): < 30 oversold, > 70 overbought, 50 = neutral.
- **ADX**: trend strength. > 25 = trending, < 20 = range-bound.
- **IV vs HV**: when IV > HV, options are expensive relative to realised moves — good for selling premium; before earnings, high IV/HV means big move priced in.
- **GEX (Gamma Exposure)**: dealer positioning. Positive GEX = dealers long gamma → price pinned near that strike. Negative GEX = dealers short gamma → price can move more violently.
- **Short float %**: shares sold short as % of float. > 15% = potential squeeze fuel.
- **Days to cover (DTC)**: days needed to cover all shorts at avg daily volume. > 5 = dangerous for shorts.
- **Squeeze score (0–100)**: composite of short float (40pts), DTC (20pts), momentum (25pts), volume ratio (15pts).
- **MA alignment**: +1 bullish (5>10>20 MA), -1 bearish, 0 mixed.
- **BB position (0–1)**: 0 = at lower band (oversold), 1 = at upper band (overbought).
- **Volume ratio**: current volume vs 5-day or 20-day average. > 1.5 = unusual activity.
- **Strategy confidence**: how aligned the options contract is with the technical signal. < 0.7 = low conviction.
- **Tier 1/2/3 squeeze**: Tier 1 = highest quality setup (score > 70), Tier 2 = moderate (50–70), Tier 3 = speculative (30–50).

## How you respond
- Be specific and actionable — reference actual numbers from the data when available.
- When no data is loaded yet, explain what the user should click to get it.
- Keep responses concise (3–8 sentences) unless the user asks for more detail.
- Never promise returns or guarantee profits. Always mention this is for research only.
- If asked about a ticker not in the data, explain that and suggest running the relevant scan.
- When explaining options strategies, always mention the max risk.
- Format numbers clearly: percentages with %, dollar values with $, scores as X/100.
"""


def is_available() -> bool:
    return _AVAILABLE


def missing_key_message() -> str:
    return (
        "**ANTHROPIC_API_KEY not set.** To enable the AI assistant:\n\n"
        "1. Get a free API key at [console.anthropic.com](https://console.anthropic.com)\n"
        "2. Add to your `.env` file: `ANTHROPIC_API_KEY=sk-ant-...`\n"
        "3. Restart the app.\n\n"
        "Cost: ~$0.001 per message with Haiku (very cheap)."
    )


# ── Context builder ────────────────────────────────────────────────────────────

def _df_to_text(df: pd.DataFrame, label: str, max_rows: int = 10) -> str:
    """Convert a DataFrame to a compact text block for injection into context."""
    if df is None or df.empty:
        return ""
    rows = df.head(max_rows)
    lines = [f"\n### {label} (top {min(len(rows), max_rows)} rows)"]
    lines.append(rows.to_string(index=False, max_cols=15))
    return "\n".join(lines)


def build_context(session_state: dict, portfolio_text: str = "") -> str:
    """
    Build the context string injected before the user's message.
    Reads from Streamlit session state to include live analysis data.
    """
    parts: list[str] = ["## Live Data Currently Loaded in the App\n"]
    found_any = False

    # ── Squeeze scan ──────────────────────────────────────────────────────────
    squeeze_df = session_state.get("squeeze_df")
    if squeeze_df is not None and not squeeze_df.empty:
        found_any = True
        parts.append(_df_to_text(
            squeeze_df[["ticker", "short_float_pct", "days_to_cover", "volume_ratio",
                         "ret_5d", "rsi", "squeeze_score", "setup_quality", "buy_timing"]
                        if all(c in squeeze_df.columns for c in ["ticker", "short_float_pct", "days_to_cover"])
                        else squeeze_df.columns[:10]],
            "Short Squeeze Scanner Results",
        ))

    # ── Strategy ideas ────────────────────────────────────────────────────────
    strategy_df = session_state.get("strategy_df")
    if strategy_df is not None and not strategy_df.empty:
        found_any = True
        strat_cols = [c for c in ["ticker", "horizon", "view", "contract_type", "expiration",
                                   "strike_price", "option_value", "strategy_score",
                                   "strategy_confidence", "entry_rule", "stop_rule"] if c in strategy_df.columns]
        parts.append(_df_to_text(strategy_df[strat_cols], "Strategy Ideas"))

    # ── Market movers ─────────────────────────────────────────────────────────
    movers_df = session_state.get("market_movers_df")
    if movers_df is not None and not movers_df.empty:
        found_any = True
        movers_cols = [c for c in ["ticker", "close", "one_week_view", "one_week_score",
                                    "one_month_view", "ret_5d", "rsi_14"] if c in movers_df.columns]
        parts.append(_df_to_text(movers_df[movers_cols], "Rapid Movers"))

    # ── Price forecast ────────────────────────────────────────────────────────
    gainers = session_state.get("forecast_gainers_df")
    if gainers is not None and not gainers.empty:
        found_any = True
        fc_cols = [c for c in ["ticker", "close", "one_week_view", "est_1w_pct",
                                "forecast_confidence", "rsi_14"] if c in gainers.columns]
        parts.append(_df_to_text(gainers[fc_cols], "Forecast Top Gainers"))

    losers = session_state.get("forecast_losers_df")
    if losers is not None and not losers.empty:
        fc_cols = [c for c in ["ticker", "close", "one_week_view", "est_1w_pct",
                                "forecast_confidence", "rsi_14"] if c in losers.columns]
        parts.append(_df_to_text(losers[fc_cols], "Forecast Top Losers"))

    # ── Earnings calendar ─────────────────────────────────────────────────────
    earnings_df = session_state.get("earnings_df")
    if earnings_df is not None and not earnings_df.empty:
        found_any = True
        earn_cols = [c for c in ["ticker", "earnings_date", "days_until", "iv_median_pct",
                                  "hv_30_pct", "iv_hv_ratio", "iv_elevated"] if c in earnings_df.columns]
        parts.append(_df_to_text(earnings_df[earn_cols], "Upcoming Earnings"))

    # ── Options chain / flow ──────────────────────────────────────────────────
    options_flow = session_state.get("options_flow_df")
    if options_flow is not None and not options_flow.empty:
        found_any = True
        parts.append(_df_to_text(options_flow.head(5), "Unusual Options Flow (last run)"))

    # ── Market screener ───────────────────────────────────────────────────────
    screener_df = session_state.get("screener_df")
    if screener_df is not None and not screener_df.empty:
        found_any = True
        parts.append(_df_to_text(screener_df.head(15), "Market Screener Results"))

    # ── Insider flow ──────────────────────────────────────────────────────────
    insider_df = session_state.get("insider_df")
    if insider_df is not None and not insider_df.empty:
        found_any = True
        ins_cols = [c for c in ["ticker", "date", "insider", "position",
                                 "transaction", "shares_fmt", "value_fmt"] if c in insider_df.columns]
        parts.append(_df_to_text(insider_df[ins_cols], "Insider Transactions"))

    # ── Single ticker analysis ────────────────────────────────────────────────
    analysis = session_state.get("analysis_result")
    if analysis:
        found_any = True
        ticker = analysis.get("ticker", "?")
        snap = analysis.get("snapshot", {})
        tech = analysis.get("technicals", {})
        fund = analysis.get("fundamentals", {})
        signal = analysis.get("signal", {})
        parts.append(
            f"\n### Ticker Analysis: {ticker}\n"
            f"Price: ${snap.get('price', '?')} | Change: {snap.get('change_pct', '?')}%\n"
            f"RSI: {tech.get('rsi_14', '?')} | ADX: {tech.get('adx_14', '?')} | "
            f"MA Alignment: {tech.get('ma_alignment', '?')}\n"
            f"Signal: {signal.get('direction', '?')} | Confidence: {signal.get('confidence', '?')}\n"
            f"P/E: {fund.get('pe_ratio', '?')} | EPS Growth: {fund.get('eps_next_y', '?')}% | "
            f"Analyst: {fund.get('analyst_recom_str', '?')} | Target: ${fund.get('target_price', '?')}"
        )

    if not found_any:
        parts.append(
            "*(No analysis data loaded yet — the user needs to run one of the scans first.)*"
        )

    # ── Portfolio ─────────────────────────────────────────────────────────────
    if portfolio_text and portfolio_text.strip():
        parts.append(f"\n## User's Portfolio / Watchlist\n{portfolio_text.strip()}")

    return "\n".join(parts)


# ── Chat API call ──────────────────────────────────────────────────────────────

def stream_response(
    messages: list[dict],
    context: str,
    model_label: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """
    Stream Claude's response token by token.
    messages = list of {"role": "user"|"assistant", "content": "..."}
    context  = build_context() output, injected as last user turn prefix
    """
    if not _AVAILABLE or _client is None:
        yield missing_key_message()
        return

    model_id = MODELS.get(model_label, MODELS[DEFAULT_MODEL])

    # Inject context into the most recent user message
    augmented = list(messages)
    if augmented and augmented[-1]["role"] == "user":
        original = augmented[-1]["content"]
        augmented[-1] = {
            "role": "user",
            "content": f"{context}\n\n---\n\n**User question:** {original}",
        }

    try:
        with _client.messages.stream(
            model=model_id,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=augmented,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n⚠️ Error: {e}"


def get_response(
    messages: list[dict],
    context: str,
    model_label: str = DEFAULT_MODEL,
) -> str:
    """Non-streaming version — returns full response string."""
    return "".join(stream_response(messages, context, model_label))
