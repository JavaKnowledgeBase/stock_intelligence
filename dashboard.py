# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:54:23 2026

@author: rkafl
"""

import ssl_fix  # must be first — patches CURL_CA_BUNDLE for yfinance + all HTTPS

import concurrent.futures
import os
import subprocess

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MARKET_SCAN_TICKERS, TICKERS
from gex_engine import calculate_gex
from options_data import (
    build_market_movers_table,
    build_price_forecast_table,
    build_strategy_table,
    build_market_volume_table,
    get_expirations,
    get_market_options_snapshot,
    get_options_chain,
)
from options_flow import detect_unusual_flow
from r2_storage import ensure_assets_available
from short_squeeze import scan_short_squeeze
from timing_engine import get_intraday_timing_batch, _SLOT_LABELS as TIMING_SLOT_LABELS
from ticker_analysis import analyse_ticker
import cache_manager as cm
from earnings_calendar import get_earnings_calendar
from screener_engine import SCREENER_PRESETS, FILTER_COLUMNS, run_screener
from insider_flow import get_insider_flow
from news_sentiment import get_news_sentiment, aggregate_sentiment
from top_movers import get_top_movers
import ai_assistant as ai


ensure_assets_available()

st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Allow tab labels to wrap onto two lines */
.stTabs [data-baseweb="tab"] {
    white-space: pre-wrap !important;
    text-align: center !important;
    height: auto !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    line-height: 1.3 !important;
    min-width: 70px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_build_label():
    env_candidates = [
        os.getenv("STREAMLIT_BUILD_COMMIT"),
        os.getenv("GITHUB_SHA"),
        os.getenv("COMMIT_SHA"),
    ]
    for value in env_candidates:
        if value:
            return value[:7]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass

    return "unknown"


st.title("Market Intelligence Platform")
st.caption(f"Branch: `main` | Build: `{get_build_label()}`")


# =========================
# CACHED DATA FETCHERS
# =========================

@st.cache_data(ttl=3600, show_spinner=False)
def get_expirations_cached(ticker):
    try:
        return get_expirations(ticker)
    except Exception:
        return []


def _fetch_snapshot(ticker, max_contracts):
    try:
        result = get_market_options_snapshot(ticker, max_contracts=max_contracts)
        return ticker, result, None
    except Exception as e:
        return ticker, pd.DataFrame(), str(e)


def fetch_all_snapshots(tickers, max_contracts, timeout=45, workers=10):
    snapshots = []
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_snapshot, t, max_contracts): t for t in tickers
        }
        done, pending = concurrent.futures.wait(futures, timeout=timeout)

        for future in done:
            ticker, result, err = future.result()
            if err:
                failed.append(ticker)
            elif not result.empty:
                snapshots.append(result)

        for future in pending:
            future.cancel()
            failed.append(futures[future])

    return snapshots, failed


# =========================
# TABLE FORMATTERS
# =========================

def format_options_table(df):
    if df is None or df.empty:
        return df

    preferred_columns = [
        "type", "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "inTheMoney", "impliedVolatility", "lastTradeDate",
    ]
    available_columns = [c for c in preferred_columns if c in df.columns]
    return df[available_columns].copy().rename(
        columns={
            "lastPrice": "Last Price",
            "openInterest": "Open Interest",
            "inTheMoney": "In The Money",
            "type": "Call or Put",
            "impliedVolatility": "Implied Volatility",
            "lastTradeDate": "Last Trade Date",
            "strike": "Strike",
            "bid": "Bid",
            "ask": "Ask",
            "volume": "Volume",
        }
    )


def format_market_volume_table(df):
    if df is None or df.empty:
        return df

    formatted_df = df[[
        "ticker", "dominant_side", "one_day_volume", "percent_of_day_total",
        "one_week_total", "one_week_trend", "call_volume", "put_volume",
    ]].copy()
    formatted_df["percent_of_day_total"] = formatted_df["percent_of_day_total"].round(2)

    return formatted_df.rename(columns={
        "ticker": "Ticker",
        "dominant_side": "Calls or Puts",
        "one_day_volume": "1 Day Volume",
        "percent_of_day_total": "% of Scanned Day Total",
        "one_week_total": "Last 1 Week Total",
        "one_week_trend": "1 Week Trend",
        "call_volume": "Call Volume",
        "put_volume": "Put Volume",
    })


def format_market_movers_table(df):
    if df is None or df.empty:
        return df

    formatted_df = df[[
        "ticker", "close", "one_week_view", "one_week_score",
        "one_month_view", "one_month_score", "ret_5d", "volatility_5d", "volume_ratio_5",
    ]].copy()

    return formatted_df.rename(columns={
        "ticker": "Ticker",
        "close": "Last Price",
        "one_week_view": "1 Week View",
        "one_week_score": "1 Week Score",
        "one_month_view": "1 Month View",
        "one_month_score": "1 Month Score",
        "ret_5d": "5 Day Return %",
        "volatility_5d": "5 Day Range %",
        "volume_ratio_5": "Volume Ratio",
    })


def format_strategy_table(df):
    if df is None or df.empty:
        return df

    base_cols = [
        "ticker", "horizon", "view", "contract_type", "expiration", "strike_price",
        "option_value", "bid", "ask", "spread_pct", "open_interest", "option_volume",
        "contract_quality_score", "strategy_score", "strategy_confidence", "day_volume_share",
        "underlying_5d_move",
    ]
    optional_cols = ["rsi_14", "adx_14", "iv_hv_ratio", "analyst_recom", "target_upside_pct"]
    text_cols = ["entry_rule", "stop_rule", "take_profit_rule", "midday_check", "daily_plan"]

    available = base_cols + [c for c in optional_cols if c in df.columns] + text_cols
    formatted_df = df[[c for c in available if c in df.columns]].copy()

    return formatted_df.rename(columns={
        "ticker": "Ticker",
        "horizon": "Horizon",
        "view": "Market View",
        "contract_type": "Call or Put",
        "expiration": "Expiration",
        "strike_price": "Strike Price",
        "option_value": "Option Value",
        "bid": "Bid",
        "ask": "Ask",
        "spread_pct": "Spread %",
        "open_interest": "Open Interest",
        "option_volume": "Option Volume",
        "contract_quality_score": "Contract Quality",
        "strategy_score": "Strategy Score",
        "strategy_confidence": "Confidence %",
        "day_volume_share": "% of Day Volume",
        "underlying_5d_move": "Underlying 5D Move %",
        "rsi_14": "RSI (14)",
        "adx_14": "ADX (14)",
        "iv_hv_ratio": "IV/HV Ratio",
        "analyst_recom": "Analyst (1-5)",
        "target_upside_pct": "Target Upside %",
        "entry_rule": "Entry Rule",
        "stop_rule": "Stop Rule",
        "take_profit_rule": "Take Profit Rule",
        "midday_check": "11 AM Check",
        "daily_plan": "Day Trader Plan",
    })


def _now():
    return pd.Timestamp.now().strftime("%Y-%m-%d %I:%M:%S %p")


# =========================
# SESSION STATE DEFAULTS
# =========================

for state_key, default_value in [
    ("market_options_raw_df", None),   # unfiltered, for live filter reapply
    ("market_options_failed", []),
    ("market_options_fetched_at", None),
    ("market_volume_df", None),
    ("market_volume_fetched_at", None),
    ("market_movers_df", None),
    ("market_movers_fetched_at", None),
    ("strategy_df", None),
    ("strategy_diagnostics", None),
    ("strategy_fetched_at", None),
    ("options_chain_df", None),
    ("options_flow_df", None),
    ("options_gex_series", None),
    ("options_chain_fetched_at", None),
    ("forecast_gainers_df", None),
    ("forecast_losers_df", None),
    ("forecast_fetched_at", None),
    ("squeeze_df", None),
    ("squeeze_fetched_at", None),
    ("timing_results", None),
    ("timing_fetched_at", None),
    ("analysis_result", None),
    ("analysis_ticker", None),
    ("analysis_fetched_at", None),
    ("earnings_df", None),
    ("earnings_fetched_at", None),
    ("screener_df", None),
    ("screener_fetched_at", None),
    ("insider_df", None),
    ("insider_fetched_at", None),
    ("news_df", None),
    ("news_agg_df", None),
    ("news_fetched_at", None),
    ("movers_gainers_df", None),
    ("movers_losers_df", None),
    ("movers_fetched_at", None),
    ("chat_history", []),
    ("chat_portfolio", ""),
    ("chat_model", ai.DEFAULT_MODEL),
    ("trader_mode", None),
]:

    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# =========================
# AI ASSISTANT SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.caption("Ask questions about any analysis loaded in the app.")

    if not ai.is_available():
        st.warning(ai.missing_key_message())
    else:
        # Model selector
        st.session_state["chat_model"] = st.selectbox(
            "Model",
            list(ai.MODELS.keys()),
            index=list(ai.MODELS.keys()).index(st.session_state["chat_model"]),
            key="chat_model_select",
        )

        # Portfolio / watchlist input
        with st.expander("My Portfolio / Watchlist (optional)", expanded=False):
            portfolio_input = st.text_area(
                "Paste your holdings (e.g. 100 NVDA @ $110, 50 TSLA @ $240)",
                value=st.session_state["chat_portfolio"],
                height=120,
                placeholder="100 NVDA @ $110\n50 TSLA @ $240\n200 SPY @ $520",
                key="portfolio_text_area",
            )
            if portfolio_input != st.session_state["chat_portfolio"]:
                st.session_state["chat_portfolio"] = portfolio_input

        # Chat input — placed after portfolio for natural flow
        col_msg, col_send = st.columns([4, 1])
        with col_msg:
            user_input = st.text_input(
                "Message",
                value="",
                placeholder="Ask about any analysis…",
                label_visibility="collapsed",
                key="chat_input_box",
            )
        with col_send:
            send_clicked = st.button("Send", use_container_width=True)

        # Clear button — above history
        col_clear, col_spacer = st.columns([1, 2])
        with col_clear:
            if st.button("Clear chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

        # Chat history display
        chat_container = st.container(height=380)
        with chat_container:
            if not st.session_state["chat_history"]:
                st.markdown(
                    "_No messages yet. Run any scan in the main tabs, "
                    "then ask me about the results._"
                )
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if send_clicked and user_input.strip():
            msg = user_input.strip()
            st.session_state["chat_history"].append(
                {"role": "user", "content": msg}
            )

            context = ai.build_context(
                st.session_state,
                portfolio_text=st.session_state["chat_portfolio"],
            )

            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in ai.stream_response(
                        st.session_state["chat_history"],
                        context,
                        model_label=st.session_state["chat_model"],
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_response}
            )
            st.rerun()

    st.divider()

    # Mode switcher
    _current_mode = st.session_state["trader_mode"]
    if _current_mode:
        _mode_label = "📈 Stocks mode" if _current_mode == "stocks" else "📊 Options mode"
        st.caption(f"**Mode:** {_mode_label}")
        if st.button("Switch mode", use_container_width=True):
            st.session_state["trader_mode"] = None
            st.rerun()
        st.divider()

    st.caption("**Platform:** Market Intelligence v3.0")
    st.caption("**Data:** yfinance · Finviz · SEC EDGAR · Polygon.io")
    if st.button("⚙️ About", use_container_width=True):
        st.info(
            "This platform provides institutional-grade market analysis for retail traders. "
            "All data is for research purposes only — not financial advice. "
            "Past signals do not guarantee future results."
        )


# =========================
# STOCK TAB RENDER FUNCTIONS
# (shared between stocks mode and options mode)
# =========================

def _render_ticker_analysis():
    st.markdown(
        "Complete single-ticker deep dive — fetches fresh data from Polygon and Finviz and "
        "produces a full technical, fundamental, timing, and strategy analysis."
    )

    col_ticker, col_run = st.columns([2, 1])
    with col_ticker:
        analysis_input = st.text_input("Ticker Symbol", value="", placeholder="e.g. NVDA").upper().strip()
    with col_run:
        st.write("")
        run_analysis = st.button("Run Full Analysis", use_container_width=True)

    if run_analysis and analysis_input:
        st.session_state["analysis_result"] = None
        st.session_state["analysis_ticker"] = analysis_input
        with st.spinner(f"Running full analysis on {analysis_input}…"):
            try:
                result = analyse_ticker(analysis_input)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_fetched_at"] = _now()
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    result = st.session_state["analysis_result"]

    if result:
        ticker = result["ticker"]
        snap = result["snapshot"]
        tech = result["technicals"]
        fund = result["fundamentals"]
        signal = result["signal"]
        timing = result["timing"]

        st.caption(f"Last run: {st.session_state['analysis_fetched_at']}")

        price = snap.get("price") or tech.get("price", 0)
        change = snap.get("change_pct") or 0
        company = fund.get("company", ticker)
        sector = fund.get("sector", "")

        st.subheader(f"{ticker} — {company}")
        st.caption(f"{sector} · {fund.get('industry', '')}")

        col_p, col_c, col_d, col_sig = st.columns(4)
        col_p.metric("Price", f"${price:.2f}")
        col_c.metric("Today", f"{change:+.2f}%")
        col_d.metric("Direction", signal["direction"])
        col_sig.metric("Confidence", f"{signal['confidence']}%")

        direction_color = (
            "green" if "Bullish" in signal["direction"] else
            "red" if "Bearish" in signal["direction"] else "gray"
        )
        st.markdown(f"### Overall Signal: :{direction_color}[{signal['direction']}]")
        st.info(signal["trade_idea"])

        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown("**Bullish signals**")
            for r in signal["bullish_reasons"]:
                st.markdown(f"- {r}")
        with col_bear:
            st.markdown("**Bearish signals**")
            for r in signal["bearish_reasons"]:
                st.markdown(f"- {r}")

        st.divider()

        with st.expander("Technical Analysis", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RSI (14)", f"{tech.get('rsi', 0):.1f}", tech.get("rsi_signal", ""))
            c2.metric("ADX (14)", f"{tech.get('adx', 0):.1f}", tech.get("adx_signal", ""))
            c3.metric("MACD", f"{tech.get('macd_hist', 0):+.3f}", tech.get("macd_signal", ""))
            c4.metric("Vol Ratio 5d", f"{tech.get('vol_ratio_5d', 0):.2f}x")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("5D Return", f"{tech.get('ret_5d', 0):+.2f}%")
            c6.metric("1M Return", f"{tech.get('ret_1mo', 0):+.2f}%")
            c7.metric("ATR (14)", f"${tech.get('atr', 0):.2f}")
            c8.metric("Trend", tech.get("trend", "—"))

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("MA 20", f"${tech.get('ma20', 0):.2f}")
            c10.metric("MA 50", f"${tech.get('ma50', 0):.2f}" if tech.get("ma50") else "—")
            c11.metric("MA 200", f"${tech.get('ma200', 0):.2f}" if tech.get("ma200") else "—")
            c12.metric("vs 52W High", f"{tech.get('pct_from_high', 0):+.1f}%")

            c13, c14 = st.columns(2)
            c13.metric("Support (30d low)", f"${tech.get('support', 0):.2f}")
            c14.metric("Resistance (30d high)", f"${tech.get('resistance', 0):.2f}")

            hist = result.get("history")
            if hist is not None and not hist.empty:
                close_col = "Close"
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.copy()
                    hist.columns = hist.columns.get_level_values(0)
                chart_df = hist[[close_col]].tail(120).copy()
                chart_df.index = pd.to_datetime(chart_df.index)
                chart_df.columns = ["Price"]
                chart_df["MA20"] = chart_df["Price"].rolling(20).mean()
                chart_df["MA50"] = chart_df["Price"].rolling(50).mean()
                fig_price = px.line(
                    chart_df, title=f"{ticker} — Last 120 Days",
                    labels={"value": "Price", "index": "Date"},
                )
                fig_price.update_layout(height=350, legend_title="")
                st.plotly_chart(fig_price, use_container_width=True)

        with st.expander("Fundamentals & Analyst Data"):
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Market Cap", fund.get("market_cap", "—"))
            f2.metric("P/E (TTM)", f"{fund.get('pe') or '—'}")
            f3.metric("Forward P/E", f"{fund.get('forward_pe') or '—'}")
            f4.metric("EPS (TTM)", f"${fund.get('eps_ttm') or '—'}")

            f5, f6, f7, f8 = st.columns(4)
            f5.metric("EPS Growth Next Y", f"{fund.get('eps_next_y') or '—'}%")
            f6.metric("Revenue Growth Y/Y", f"{fund.get('revenue_growth') or '—'}%")
            f7.metric("Profit Margin", f"{fund.get('profit_margin') or '—'}%")
            f8.metric("ROE", f"{fund.get('roe') or '—'}%")

            f9, f10, f11, f12 = st.columns(4)
            f9.metric("Beta", f"{fund.get('beta') or '—'}")
            f10.metric("Debt/Equity", f"{fund.get('debt_eq') or '—'}")
            f11.metric("Inst. Ownership", f"{fund.get('inst_own') or '—'}%")
            f12.metric("Insider Trans", f"{fund.get('insider_trans') or '—'}%")

            fa, fb, fc, fd = st.columns(4)
            fa.metric("Analyst", fund.get("analyst_recom_str", "—"))
            fb.metric("Price Target", f"${fund.get('target_price') or '—'}")
            fc.metric("Target Upside", f"{fund.get('target_upside_pct') or '—'}%")
            fd.metric("Earnings Date", fund.get("earnings_date", "—"))

            fe, ff = st.columns(2)
            fe.metric("Short Float %", f"{fund.get('short_float_pct') or '—'}%")
            ff.metric("Short Ratio", f"{fund.get('short_ratio') or '—'}")

        with st.expander("Intraday Timing"):
            if timing.get("error"):
                st.warning(f"Timing unavailable: {timing['error']}")
            else:
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Best Entry", timing.get("best_entry_window", "—"))
                tc2.metric("Best Exit", timing.get("best_exit_window", "—"))
                tc3.metric("Gap Fade Prob", f"{timing.get('gap_fade_prob') or '—'}%")
                st.info(timing.get("timing_note", ""))

                if timing.get("slot_labels") and timing.get("avg_return_by_slot"):
                    slot_df = pd.DataFrame({
                        "Time Slot": timing["slot_labels"],
                        "Avg Return %": timing["avg_return_by_slot"],
                        "Win Rate %": timing["win_rate_by_slot"],
                        "Vol vs Avg": timing["volume_profile"],
                    })
                    entry_label = timing["best_entry_window"]
                    exit_label = timing["best_exit_window"]

                    def _hl(row):
                        if row["Time Slot"] == entry_label:
                            return ["background-color: #1a472a; color: white"] * len(row)
                        if row["Time Slot"] == exit_label:
                            return ["background-color: #1a3a5c; color: white"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        slot_df.style.apply(_hl, axis=1),
                        use_container_width=True, hide_index=True,
                    )

        st.caption(
            f"Data sources: Polygon.io (price, bars) · Finviz (fundamentals, analyst) · "
            f"yfinance (options). Analysis as of {st.session_state['analysis_fetched_at']}."
        )

    else:
        st.info("Enter a ticker symbol above and click **Run Full Analysis**.")


def _render_earnings():
    st.markdown(
        "Upcoming earnings events with **Implied Volatility context** — "
        "IV/HV ratio ≥ 1.25 means options are pricing in a bigger move than historical volatility suggests. "
        "Great for finding elevated-premium setups before announcements."
    )

    col_run, col_info, col_days = st.columns([1, 2, 1])
    with col_run:
        run_earnings = st.button("Run Earnings Calendar", use_container_width=True)
    with col_info:
        if st.session_state["earnings_fetched_at"]:
            st.caption(f"Last run: {st.session_state['earnings_fetched_at']}")
    with col_days:
        earnings_days_ahead = st.slider("Days Ahead", min_value=7, max_value=60, value=30)

    if run_earnings:
        cached_earn = cm.load("earnings_calendar")
        if cached_earn:
            st.session_state["earnings_df"] = pd.DataFrame(cached_earn)
            st.session_state["earnings_fetched_at"] = f"cached · {cm.age_label('earnings_calendar')}"
        else:
            with st.spinner(f"Fetching earnings dates for {len(MARKET_SCAN_TICKERS)} tickers..."):
                try:
                    earn_df = get_earnings_calendar(
                        MARKET_SCAN_TICKERS, max_days_ahead=earnings_days_ahead
                    )
                    st.session_state["earnings_df"] = earn_df
                    st.session_state["earnings_fetched_at"] = _now()
                    if earn_df is not None and not earn_df.empty:
                        cm.save("earnings_calendar", earn_df.to_dict("records"), ttl_seconds=3600)
                except Exception as e:
                    st.error(f"Earnings calendar failed: {e}")

    earnings_df = st.session_state["earnings_df"]

    if earnings_df is not None and not earnings_df.empty:
        display_earn = earnings_df[earnings_df["days_until"] <= earnings_days_ahead].copy()

        _earn_col_rename = {
            "ticker": "Ticker",
            "earnings_date": "Earnings Date",
            "days_until": "Days Until",
            "eps_estimate": "EPS Est.",
            "rev_estimate_b": "Rev Est. ($B)",
            "iv_median_pct": "IV Median %",
            "hv_30_pct": "HV 30d %",
            "iv_hv_ratio": "IV/HV Ratio",
            "iv_elevated": "IV Elevated?",
            "price": "Price",
            "market_cap_b": "Mkt Cap ($B)",
            "sector": "Sector",
        }
        available_earn = {k: v for k, v in _earn_col_rename.items() if k in display_earn.columns}
        st.subheader(f"Earnings in the Next {earnings_days_ahead} Days")

        def _style_earnings(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "IV Elevated?" in cols:
                idx = cols.index("IV Elevated?")
                if row["IV Elevated?"] is True or str(row["IV Elevated?"]).lower() == "true":
                    styles[idx] = "background-color: #2d1b1b; color: #ff6b6b"
            if "Days Until" in cols:
                idx = cols.index("Days Until")
                try:
                    d = int(row["Days Until"])
                    if d <= 3:
                        styles[idx] = "background-color: #2d1b00; color: #ffaa00; font-weight: bold"
                    elif d <= 7:
                        styles[idx] = "background-color: #1a2d00; color: #88cc44"
                except Exception:
                    pass
            return styles

        display_earn_renamed = display_earn[list(available_earn.keys())].rename(columns=available_earn)
        st.dataframe(
            display_earn_renamed.style.apply(_style_earnings, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        if "iv_hv_ratio" in display_earn.columns and "ticker" in display_earn.columns:
            iv_chart_df = display_earn[display_earn["iv_hv_ratio"].notna()].copy()
            if not iv_chart_df.empty:
                fig_earn = px.bar(
                    iv_chart_df.head(20),
                    x="ticker",
                    y="iv_hv_ratio",
                    color="iv_elevated",
                    title="IV/HV Ratio by Ticker (red = elevated premium)",
                    labels={"ticker": "Ticker", "iv_hv_ratio": "IV / HV Ratio", "iv_elevated": "IV Elevated"},
                    color_discrete_map={True: "#e05c5c", False: "#4a9eff"},
                )
                fig_earn.add_hline(y=1.25, line_dash="dot", line_color="orange",
                                   annotation_text="1.25× threshold")
                st.plotly_chart(fig_earn, use_container_width=True)

        st.caption(
            "IV Median % = median implied volatility across nearest expiry call options. "
            "HV 30d % = 30-day realised historical volatility (annualised). "
            "IV/HV ≥ 1.25 (red) means the market is pricing in a larger move than recent history — "
            "often driven by earnings uncertainty. Data via yfinance, updated hourly."
        )

    elif earnings_df is not None:
        st.warning("No upcoming earnings found for the configured ticker universe.")
    else:
        st.info("Click **Run Earnings Calendar** to load upcoming events.")


def _render_market_screener():
    st.markdown(
        "Filter the entire market universe by any combination of **technical factors**. "
        "Uses pre-computed daily features (RSI, MACD, momentum, volatility, MA alignment, volume ratios). "
        "Choose a preset or build your own filters."
    )

    col_preset, col_spacer = st.columns([2, 2])
    with col_preset:
        preset_choice = st.selectbox(
            "Quick Preset",
            ["Custom"] + list(SCREENER_PRESETS.keys()),
        )

    if preset_choice != "Custom":
        preset_info = SCREENER_PRESETS[preset_choice]
        st.info(f"**{preset_choice}:** {preset_info['description']}")
        preset_filters = preset_info["filters"]
    else:
        preset_filters = {}

    with st.expander("Custom Filters (override preset values)", expanded=(preset_choice == "Custom")):
        _filter_cols = st.columns(3)
        custom_filters: dict = {}

        filter_defs = [
            ("rsi_14",           "RSI (14)",            0.0,  100.0, 0.0,  100.0, 1.0),
            ("close_ret_5d",     "5D Return %",         -30.0, 50.0, -30.0, 50.0, 0.5),
            ("volume_ratio_5",   "Vol Ratio 5d",        0.0,   5.0,  0.0,   5.0,  0.1),
            ("pct_from_52w_high","% from 52W High",     -80.0, 20.0, -80.0, 20.0, 1.0),
            ("dist_ma_20_pct",   "Dist from 20MA %",    -30.0, 30.0, -30.0, 30.0, 0.5),
            ("adx_14",           "ADX (14)",             0.0,  80.0,  0.0,  80.0, 1.0),
        ]

        for i, (col_key, label, abs_min, abs_max, default_min, default_max, step) in enumerate(filter_defs):
            with _filter_cols[i % 3]:
                preset_val = preset_filters.get(col_key, {})
                d_min = preset_val.get("min", default_min)
                d_max = preset_val.get("max", default_max)
                low, high = st.slider(
                    label,
                    min_value=float(abs_min),
                    max_value=float(abs_max),
                    value=(float(d_min), float(d_max)),
                    step=float(step),
                )
                if low > abs_min or high < abs_max:
                    custom_filters[col_key] = {}
                    if low > abs_min:
                        custom_filters[col_key]["min"] = low
                    if high < abs_max:
                        custom_filters[col_key]["max"] = high

    active_filters = {**preset_filters, **custom_filters}

    col_run_scr, col_sort, col_info_scr = st.columns([1, 1, 2])
    with col_run_scr:
        run_screener_btn = st.button("Run Screener", use_container_width=True)
    with col_sort:
        sort_col = st.selectbox("Sort By", list(FILTER_COLUMNS.keys()), index=0)
    with col_info_scr:
        if st.session_state["screener_fetched_at"]:
            st.caption(f"Last run: {st.session_state['screener_fetched_at']}")

    if run_screener_btn:
        _prog = st.progress(0, text=f"Bulk downloading price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
        try:
            import data_fetcher as _df
            _df.bulk_preload(MARKET_SCAN_TICKERS, period="1y")
            _prog.progress(40, text="Price data ready — computing technical indicators & applying filters…")
            screener_result = run_screener(
                MARKET_SCAN_TICKERS,
                active_filters,
                sort_by=sort_col,
                ascending=(sort_col == "rsi_14"),
            )
            _prog.progress(100, text="Done!")
            _prog.empty()
            st.session_state["screener_df"] = screener_result
            st.session_state["screener_fetched_at"] = _now()
        except Exception as e:
            _prog.empty()
            st.error(f"Screener failed: {e}")

    screener_df = st.session_state["screener_df"]

    if screener_df is not None and not screener_df.empty:
        rename_map = {"ticker": "Ticker", "close": "Price"}
        rename_map.update({k: v for k, v in FILTER_COLUMNS.items() if k in screener_df.columns})
        st.subheader(f"{len(screener_df)} tickers matched")
        st.dataframe(
            screener_df.rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )

        if "rsi_14" in screener_df.columns and "close_ret_5d" in screener_df.columns:
            fig_scr = px.scatter(
                screener_df,
                x="rsi_14",
                y="close_ret_5d",
                color="ma_alignment" if "ma_alignment" in screener_df.columns else None,
                hover_name="ticker",
                text="ticker",
                title="RSI vs 5D Return (by MA alignment)",
                labels={"rsi_14": "RSI (14)", "close_ret_5d": "5D Return %", "ma_alignment": "MA Align"},
            )
            fig_scr.update_traces(textposition="top center")
            fig_scr.add_vline(x=30, line_dash="dot", line_color="green", annotation_text="Oversold")
            fig_scr.add_vline(x=70, line_dash="dot", line_color="red", annotation_text="Overbought")
            fig_scr.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_scr, use_container_width=True)

        st.caption(
            "Features are loaded from pre-computed daily snapshots (data/features/). "
            "Tickers without a snapshot are fetched live from yfinance. "
            "MA Alignment: +1 = bullish stack (5>10>20), -1 = bearish, 0 = mixed."
        )

    elif screener_df is not None:
        st.warning("No tickers matched the current filters. Try widening the ranges.")
    else:
        st.info("Select a preset or set custom filters, then click **Run Screener**.")


def _render_insider_flow():
    st.markdown(
        "Recent **insider buy/sell transactions** from Form 4 SEC filings. "
        "C-suite and director buys — especially large open-market purchases — "
        "are historically bullish signals. Large insider sells in clusters are bearish."
    )

    col_run_ins, col_txn_filter, col_rows_ins, col_info_ins = st.columns([1, 1, 1, 2])
    with col_run_ins:
        run_insider = st.button("Run Insider Flow", use_container_width=True)
    with col_txn_filter:
        insider_txn_filter = st.selectbox("Transaction Type", ["All", "Buy", "Sale"])
    with col_rows_ins:
        insider_per_ticker = st.slider("Per Ticker", min_value=1, max_value=10, value=3)
    with col_info_ins:
        if st.session_state["insider_fetched_at"]:
            st.caption(f"Last run: {st.session_state['insider_fetched_at']}")

    insider_ticker_input = st.text_input(
        "Filter Tickers (comma-separated, or leave blank for full universe)",
        value="",
        placeholder="e.g. NVDA, TSLA, AAPL",
    )

    if run_insider:
        raw_ins = insider_ticker_input.strip()
        ins_tickers = (
            [t.strip().upper() for t in raw_ins.split(",") if t.strip()]
            if raw_ins else MARKET_SCAN_TICKERS
        )
        with st.spinner(f"Fetching insider transactions for {len(ins_tickers)} tickers…"):
            try:
                insider_df = get_insider_flow(
                    ins_tickers,
                    max_per_ticker=insider_per_ticker,
                    transaction_filter=insider_txn_filter,
                )
                st.session_state["insider_df"] = insider_df
                st.session_state["insider_fetched_at"] = _now()
            except Exception as e:
                st.error(f"Insider flow failed: {e}")

    insider_df = st.session_state["insider_df"]

    if insider_df is not None and not insider_df.empty:
        display_ins = insider_df.copy()
        if insider_txn_filter != "All":
            display_ins = display_ins[display_ins["transaction"] == insider_txn_filter]

        _ins_col_rename = {
            "ticker": "Ticker",
            "date": "Date",
            "insider": "Insider",
            "position": "Title",
            "transaction": "Type",
            "transaction_detail": "Transaction Detail",
            "shares_fmt": "Shares",
            "value_fmt": "Value (USD)",
        }
        avail_ins = {k: v for k, v in _ins_col_rename.items() if k in display_ins.columns}

        def _style_insider(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "Type" in cols:
                idx = cols.index("Type")
                if str(row["Type"]) == "Buy":
                    styles[idx] = "background-color: #1a2d00; color: #88cc44; font-weight: bold"
                elif str(row["Type"]) == "Sale":
                    styles[idx] = "background-color: #2d1b1b; color: #ff6b6b; font-weight: bold"
            return styles

        st.subheader(f"{len(display_ins)} insider transactions")
        st.dataframe(
            display_ins[list(avail_ins.keys())].rename(columns=avail_ins).style.apply(_style_insider, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        if "transaction" in insider_df.columns and "ticker" in insider_df.columns:
            txn_counts = (
                insider_df[insider_df["transaction"].isin(["Buy", "Sale"])]
                .groupby(["ticker", "transaction"])
                .size()
                .reset_index(name="count")
            )
            if not txn_counts.empty:
                top_tickers = (
                    txn_counts.groupby("ticker")["count"].sum()
                    .nlargest(20).index.tolist()
                )
                fig_ins = px.bar(
                    txn_counts[txn_counts["ticker"].isin(top_tickers)],
                    x="ticker",
                    y="count",
                    color="transaction",
                    barmode="group",
                    title="Insider Buy vs Sale Count (top tickers)",
                    color_discrete_map={"Buy": "#88cc44", "Sale": "#e05c5c"},
                    labels={"ticker": "Ticker", "count": "# Transactions", "transaction": "Type"},
                )
                st.plotly_chart(fig_ins, use_container_width=True)

        if "edgar_link" in display_ins.columns and "ticker" in display_ins.columns:
            with st.expander("SEC EDGAR Form 4 Filing Links"):
                seen_links: set = set()
                for _, row in display_ins.iterrows():
                    link = row.get("edgar_link", "")
                    ticker_val = row.get("ticker", "")
                    if link and ticker_val not in seen_links:
                        st.markdown(f"- [{ticker_val} — Form 4 filings]({link})")
                        seen_links.add(ticker_val)

        st.caption(
            "Data from Yahoo Finance (insider_transactions). "
            "Form 4 filings are required within 2 business days of the transaction. "
            "Buy = open-market purchase. Sale = open-market disposal. "
            "Always cross-reference with SEC EDGAR for full details."
        )

    elif insider_df is not None:
        st.warning("No insider transactions found for the selected tickers / filter.")
    else:
        st.info("Click **Run Insider Flow** to load recent Form 4 transactions.")


def _render_news_sentiment():
    st.markdown(
        "Scan recent headlines across your chosen tickers and score each one as "
        "**Bullish**, **Bearish**, or **Neutral** using keyword analysis. "
        "Quickly spot which stocks are in positive vs negative news flow."
    )

    col_tickers, col_max, col_run = st.columns([3, 1, 1])
    with col_tickers:
        news_ticker_input = st.text_input(
            "Tickers (comma-separated, or leave blank for full universe)",
            value="",
            placeholder="e.g. NVDA, AAPL, MSFT, TSLA",
        )
    with col_max:
        headlines_per_ticker = st.slider("Headlines each", min_value=3, max_value=15, value=5)
    with col_run:
        st.write("")
        run_news = st.button("Scan News", use_container_width=True)

    if st.session_state["news_fetched_at"]:
        st.caption(f"Last run: {st.session_state['news_fetched_at']}")

    if run_news:
        raw = news_ticker_input.strip()
        scan_tickers = (
            [t.strip().upper() for t in raw.split(",") if t.strip()]
            if raw else MARKET_SCAN_TICKERS[:30]
        )
        with st.spinner(f"Fetching headlines for {len(scan_tickers)} tickers…"):
            try:
                news_df = get_news_sentiment(scan_tickers, max_headlines_per_ticker=headlines_per_ticker)
                st.session_state["news_df"] = news_df
                st.session_state["news_agg_df"] = aggregate_sentiment(news_df)
                st.session_state["news_fetched_at"] = _now()
            except Exception as e:
                st.error(f"News scan failed: {e}")

    news_df = st.session_state["news_df"]
    agg_df = st.session_state["news_agg_df"]

    if agg_df is not None and not agg_df.empty:
        st.subheader("Sentiment Summary by Ticker")

        def _style_agg(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "overall" in cols:
                idx = cols.index("overall")
                if row["overall"] == "Bullish":
                    styles[idx] = "background-color: #1a2d00; color: #88cc44; font-weight: bold"
                elif row["overall"] == "Bearish":
                    styles[idx] = "background-color: #2d1b1b; color: #ff6b6b; font-weight: bold"
            return styles

        agg_rename = {
            "ticker": "Ticker", "headlines": "Headlines",
            "avg_score": "Avg Score", "bullish": "Bullish",
            "neutral": "Neutral", "bearish": "Bearish", "overall": "Overall",
        }
        st.dataframe(
            agg_df.rename(columns=agg_rename).style.apply(_style_agg, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Score bar chart
        fig_news = px.bar(
            agg_df,
            x="ticker",
            y="avg_score",
            color="overall",
            title="Average Sentiment Score by Ticker",
            labels={"ticker": "Ticker", "avg_score": "Avg Score", "overall": "Sentiment"},
            color_discrete_map={"Bullish": "#88cc44", "Neutral": "#888888", "Bearish": "#e05c5c"},
        )
        fig_news.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_news, use_container_width=True)

    if news_df is not None and not news_df.empty:
        st.subheader("All Headlines")
        sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Bullish", "Neutral", "Bearish"])
        display_news = news_df if sentiment_filter == "All" else news_df[news_df["sentiment"] == sentiment_filter]

        def _style_news(row):
            styles = [""] * len(row)
            cols = list(row.index)
            if "Sentiment" in cols:
                idx = cols.index("Sentiment")
                if row["Sentiment"] == "Bullish":
                    styles[idx] = "background-color: #1a2d00; color: #88cc44"
                elif row["Sentiment"] == "Bearish":
                    styles[idx] = "background-color: #2d1b1b; color: #ff6b6b"
            return styles

        news_rename = {
            "ticker": "Ticker", "date": "Date", "title": "Headline",
            "source": "Source", "score": "Score", "sentiment": "Sentiment",
        }
        cols_to_show = [c for c in ["ticker", "date", "title", "source", "score", "sentiment"] if c in display_news.columns]
        st.dataframe(
            display_news[cols_to_show].rename(columns=news_rename).style.apply(_style_news, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Sentiment scored by counting bullish vs bearish keywords in the headline. "
            "Score +1 = all bullish words, -1 = all bearish, 0 = mixed or neutral. "
            "Headlines via Finviz. Not a substitute for reading the full article."
        )

    elif news_df is None:
        st.info("Enter tickers (or leave blank for top 30 of universe) and click **Scan News**.")


def _render_top_movers():
    st.markdown(
        "Today's biggest **price movers** across the ticker universe — "
        "ranked by percentage change from yesterday's close."
    )

    col_run, col_n, col_info = st.columns([1, 1, 2])
    with col_run:
        run_movers = st.button("Refresh Movers", use_container_width=True)
    with col_n:
        top_n = st.slider("Top N", min_value=5, max_value=25, value=10)
    with col_info:
        if st.session_state["movers_fetched_at"]:
            st.caption(f"Last run: {st.session_state['movers_fetched_at']}")

    if run_movers:
        with st.spinner(f"Fetching price data for {len(MARKET_SCAN_TICKERS)} tickers…"):
            try:
                gainers, losers = get_top_movers(MARKET_SCAN_TICKERS, top_n=top_n)
                st.session_state["movers_gainers_df"] = gainers
                st.session_state["movers_losers_df"] = losers
                st.session_state["movers_fetched_at"] = _now()
            except Exception as e:
                st.error(f"Top movers failed: {e}")

    gainers = st.session_state["movers_gainers_df"]
    losers = st.session_state["movers_losers_df"]

    _movers_rename = {
        "ticker": "Ticker", "price": "Price", "prev_close": "Prev Close",
        "change_pct": "Change %", "volume": "Avg 3M Vol",
    }

    if gainers is not None and not gainers.empty:
        col_g, col_l = st.columns(2)

        with col_g:
            st.subheader("🟢 Top Gainers")
            st.dataframe(
                gainers.head(top_n).rename(columns=_movers_rename),
                use_container_width=True,
                hide_index=True,
            )

        with col_l:
            st.subheader("🔴 Top Losers")
            if losers is not None and not losers.empty:
                st.dataframe(
                    losers.head(top_n).rename(columns=_movers_rename),
                    use_container_width=True,
                    hide_index=True,
                )

        # Combined bar chart
        if gainers is not None and losers is not None:
            top_g = gainers.head(10).copy()
            top_l = losers.head(10).copy()
            chart_df = pd.concat([top_g, top_l]).sort_values("change_pct", ascending=False)
            fig_mov = px.bar(
                chart_df,
                x="ticker",
                y="change_pct",
                color="change_pct",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Today's Top Movers — % Change",
                labels={"ticker": "Ticker", "change_pct": "% Change"},
                text="change_pct",
            )
            fig_mov.update_traces(texttemplate="%{text:+.2f}%", textposition="outside")
            fig_mov.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_mov.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_mov, use_container_width=True)

        st.caption(
            "% change calculated from previous close via yfinance fast_info. "
            "Data may lag intraday by a few minutes. Run again to refresh."
        )

    elif gainers is None:
        st.info("Click **Refresh Movers** to load today's price movements.")


# =========================
# MODE GATE — welcome screen or tabs
# =========================

if st.session_state["trader_mode"] is None:

    st.markdown("---")
    st.markdown("## What kind of trader are you?")
    st.markdown("Choose your focus area. You can switch anytime from the sidebar.")
    st.write("")

    col_stocks, col_options = st.columns(2)

    with col_stocks:
        st.markdown("### 📈 Stock Trader")
        st.markdown(
            "Research and scan individual stocks with:\n"
            "- **Ticker Analysis** — technical, fundamental & timing deep dive\n"
            "- **Earnings Calendar** — upcoming events with IV context\n"
            "- **Market Screener** — filter by RSI, momentum, MA alignment\n"
            "- **Insider Flow** — Form 4 SEC filings (C-suite buys & sells)\n"
            "- **News Sentiment** — bullish/bearish headline scanner\n"
            "- **Top Movers** — today's biggest gainers & losers\n"
        )
        st.write("")
        if st.button("I trade Stocks →", use_container_width=True, type="primary"):
            st.session_state["trader_mode"] = "stocks"
            st.rerun()

    with col_options:
        st.markdown("### 📊 Options Trader")
        st.markdown(
            "Institutional-grade options intelligence with:\n"
            "- **Options Screener** — scan calls & puts across the market\n"
            "- **Volume Leaders** — biggest options activity today\n"
            "- **Strategy Ideas** — ML-ranked trade setups with entry/exit rules\n"
            "- **Chain Explorer**, **Price Forecast**, **Short Squeeze**, **Intraday Timing**\n"
        )
        st.write("")
        if st.button("I trade Options →", use_container_width=True, type="primary"):
            st.session_state["trader_mode"] = "options"
            st.rerun()


# =========================
# STOCKS MODE
# =========================

elif st.session_state["trader_mode"] == "stocks":

    stab1, stab2, stab3, stab4, stab5, stab6 = st.tabs([
        "Ticker\nAnalysis",
        "Earnings\nCalendar",
        "Market\nScreener",
        "Insider\nFlow",
        "News\nSentiment",
        "Top\nMovers",
    ])

    with stab1:
        _render_ticker_analysis()

    with stab2:
        _render_earnings()

    with stab3:
        _render_market_screener()

    with stab4:
        _render_insider_flow()

    with stab5:
        _render_news_sentiment()

    with stab6:
        _render_top_movers()


# =========================
# OPTIONS MODE
# =========================

elif st.session_state["trader_mode"] == "options":

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Options\nScreener",
        "Volume\nLeaders",
        "Rapid\nMovers",
        "Strategy\nIdeas",
        "Chain\nExplorer",
        "Price\nForecast",
        "Short\nSqueeze",
        "Intraday\nTiming",
        "Ticker\nAnalysis",
    ])

    # =========================
    # MARKET OPTIONS SCREENER
    # =========================

    with tab1:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_market_options = st.button("Run Market Options Screener", use_container_width=True)
        with col_info:
            if st.session_state["market_options_fetched_at"]:
                failed = st.session_state["market_options_failed"]
                fail_msg = f" | {len(failed)} tickers failed" if failed else ""
                st.caption(f"Last run: {st.session_state['market_options_fetched_at']}{fail_msg}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            contract_type_filter = st.selectbox("Call or Put", ["All", "call", "put"])
        with col2:
            itm_filter = st.selectbox("In The Money", ["All", "Yes", "No"])
        with col3:
            min_volume = st.number_input("Minimum Volume", min_value=0, value=1, step=1)
        with col4:
            contracts_per_ticker = st.slider("Contracts Per Ticker", min_value=3, max_value=20, value=8)

        if run_market_options:
            with st.spinner(f"Fetching contracts for {len(TICKERS)} tickers (parallel)..."):
                snapshots, failed = fetch_all_snapshots(TICKERS, contracts_per_ticker)

            st.session_state["market_options_fetched_at"] = _now()
            st.session_state["market_options_failed"] = failed

            if snapshots:
                st.session_state["market_options_raw_df"] = pd.concat(snapshots, ignore_index=True)
            else:
                st.session_state["market_options_raw_df"] = pd.DataFrame()

            if failed:
                st.warning(f"{len(failed)} tickers had no data or timed out: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

        raw_df = st.session_state["market_options_raw_df"]

        if raw_df is not None and not raw_df.empty:
            filtered_df = raw_df[raw_df["volume"] >= min_volume].copy()

            if contract_type_filter != "All":
                filtered_df = filtered_df[filtered_df["option_type"] == contract_type_filter]

            if itm_filter == "Yes":
                filtered_df = filtered_df[filtered_df["in_the_money"]]
            elif itm_filter == "No":
                filtered_df = filtered_df[~filtered_df["in_the_money"]]

            filtered_df = filtered_df.sort_values(
                ["volume", "open_interest", "last_price"],
                ascending=[False, False, False],
            )

            display_df = filtered_df.rename(columns={
                "ticker": "Ticker",
                "expiration": "Expiration",
                "option_type": "Call or Put",
                "strike": "Strike",
                "last_price": "Last Price",
                "bid": "Bid",
                "ask": "Ask",
                "volume": "Volume",
                "open_interest": "Open Interest",
                "in_the_money": "In The Money",
            })

            st.caption(f"Showing {len(display_df)} contracts from {raw_df['ticker'].nunique() if 'ticker' in raw_df.columns else '?'} tickers")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if not filtered_df.empty:
                fig = px.scatter(
                    filtered_df,
                    x="strike",
                    y="last_price",
                    size="volume",
                    color="option_type",
                    hover_name="ticker",
                    hover_data=["expiration", "bid", "ask", "open_interest"],
                    title="Option Contract Activity",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif raw_df is not None:
            st.info("No contracts matched your filters.")
        else:
            st.info("Click Run to load contracts.")


    # =========================
    # MARKET VOLUME LEADERS
    # =========================

    with tab2:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_market_volume = st.button("Run Market Volume Leaders", use_container_width=True)
        with col_info:
            if st.session_state["market_volume_fetched_at"]:
                st.caption(f"Last run: {st.session_state['market_volume_fetched_at']}")

        volume_leaders_count = st.slider("Volume Leaders Rows", min_value=10, max_value=50, value=20)

        if run_market_volume:
            cached = cm.load("market_volume")
            if cached:
                st.session_state["market_volume_df"] = pd.DataFrame(cached)
                st.session_state["market_volume_fetched_at"] = f"cached · {cm.age_label('market_volume')}"
            else:
                _prog = st.progress(0, text=f"Bulk downloading price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
                try:
                    import data_fetcher as _df
                    _df.bulk_preload(MARKET_SCAN_TICKERS, period="1y")
                    _prog.progress(40, text="Price data ready — scanning volume leaders with 50 workers…")
                    result = build_market_volume_table(MARKET_SCAN_TICKERS)
                    _prog.progress(100, text="Done!")
                    _prog.empty()
                    st.session_state["market_volume_df"] = result
                    st.session_state["market_volume_fetched_at"] = _now()
                    if result is not None and not result.empty:
                        cm.save("market_volume", result.to_dict("records"))
                except Exception as e:
                    _prog.empty()
                    st.error(f"Failed to load volume data: {e}")

        market_volume_df = st.session_state["market_volume_df"]

        if market_volume_df is not None and not market_volume_df.empty:
            st.dataframe(
                format_market_volume_table(market_volume_df.head(volume_leaders_count)),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Free-data approximation using the scanned symbol universe and the "
                "nearest expiry option chain. Weekly totals and trend come from saved "
                "daily snapshots under data/options_market_snapshots."
            )
        elif market_volume_df is not None:
            st.warning("No market volume data was returned.")
        else:
            st.info("Click Run to load volume leaders.")


    # =========================
    # RAPID MOVERS
    # =========================

    with tab3:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_market_movers = st.button("Run Rapid Movers", use_container_width=True)
        with col_info:
            if st.session_state["market_movers_fetched_at"]:
                st.caption(f"Last run: {st.session_state['market_movers_fetched_at']}")

        movers_row_count = st.slider("Mover Rows", min_value=10, max_value=20, value=10)

        if run_market_movers:
            cached = cm.load("market_movers")
            if cached:
                st.session_state["market_movers_df"] = pd.DataFrame(cached)
                st.session_state["market_movers_fetched_at"] = f"cached · {cm.age_label('market_movers')}"
            else:
                _prog = st.progress(0, text=f"Bulk downloading price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
                try:
                    import data_fetcher as _df
                    _df.bulk_preload(MARKET_SCAN_TICKERS, period="1y")
                    _prog.progress(40, text="Price data ready — identifying rapid movers with 50 workers…")
                    result = build_market_movers_table(MARKET_SCAN_TICKERS)
                    _prog.progress(100, text="Done!")
                    _prog.empty()
                    st.session_state["market_movers_df"] = result
                    st.session_state["market_movers_fetched_at"] = _now()
                    if result is not None and not result.empty:
                        cm.save("market_movers", result.to_dict("records"))
                except Exception as e:
                    _prog.empty()
                    st.error(f"Failed to load movers data: {e}")

        movers_df = st.session_state["market_movers_df"]

        if movers_df is not None and not movers_df.empty:
            st.dataframe(
                format_market_movers_table(movers_df.head(movers_row_count)),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Free-data daily candidates, recalculated from fresh price, range, momentum, "
                "and volume behavior across the scanned market universe."
            )
        elif movers_df is not None:
            st.warning("No rapid mover candidates were returned.")
        else:
            st.info("Click Run to scan for movers.")


    # =========================
    # STRATEGY TABLE
    # =========================

    with tab4:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_strategy = st.button("Run Strategy", use_container_width=True)
        with col_info:
            if st.session_state["strategy_fetched_at"]:
                st.caption(f"Last run: {st.session_state['strategy_fetched_at']}")

        strategy_row_count = st.slider("Strategy Rows", min_value=5, max_value=15, value=10)

        if run_strategy:
            st.session_state["strategy_df"] = None
            st.session_state["strategy_diagnostics"] = None

            cached_strat = cm.load("strategy")
            cached_diag = cm.load("strategy_diagnostics")
            if cached_strat and cached_diag:
                st.session_state["strategy_df"] = pd.DataFrame(cached_strat)
                st.session_state["strategy_diagnostics"] = cached_diag
                st.session_state["strategy_fetched_at"] = f"cached · {cm.age_label('strategy')}"
            else:
                _prog = st.progress(0, text=f"Bulk downloading price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
                try:
                    import data_fetcher as _df
                    _df.bulk_preload(MARKET_SCAN_TICKERS, period="1y")
                    _prog.progress(40, text="Price data ready — building strategy signals with 50 workers…")
                    strategy_df, strategy_diagnostics = build_strategy_table(
                        MARKET_SCAN_TICKERS,
                        top_n=strategy_row_count,
                    )
                    _prog.progress(100, text="Done!")
                    _prog.empty()
                    st.session_state["strategy_diagnostics"] = strategy_diagnostics
                    st.session_state["strategy_df"] = (
                        pd.DataFrame() if strategy_diagnostics["status"] == "data_unavailable"
                        else strategy_df
                    )
                    st.session_state["strategy_fetched_at"] = _now()
                    if strategy_df is not None and not strategy_df.empty:
                        cm.save("strategy", strategy_df.to_dict("records"))
                        cm.save("strategy_diagnostics", strategy_diagnostics)
                except Exception as e:
                    _prog.empty()
                    st.warning(f"Strategy scan encountered an error: {e}. Showing any partial results.")

        strategy_df = st.session_state["strategy_df"]
        strategy_diagnostics = st.session_state["strategy_diagnostics"]

        if strategy_diagnostics is not None:
            freshness_label = (
                "Fresh data" if strategy_diagnostics["status"] != "data_unavailable"
                else "Stale data blocked"
            )
            regime = strategy_diagnostics.get("regime", "neutral")
            st.caption(
                f"{freshness_label} | Status: {strategy_diagnostics['status']} | "
                f"Market Regime: {regime.capitalize()} | "
                f"Tickers: {strategy_diagnostics['tickers_requested']} | "
                f"Movers: {strategy_diagnostics['movers_found']} | "
                f"Volume rows: {strategy_diagnostics['volume_rows_found']} | "
                f"Candidates: {strategy_diagnostics['combined_candidates']} | "
                f"Evaluated: {strategy_diagnostics['contracts_evaluated']} | "
                f"Selected: {strategy_diagnostics['contracts_selected']}"
            )

            if strategy_diagnostics.get("message"):
                if strategy_diagnostics["status"] in {"data_unavailable", "no_contracts"}:
                    st.warning(strategy_diagnostics["message"])
                elif strategy_diagnostics["status"] == "partial_results":
                    st.info(strategy_diagnostics["message"])

        if (
            strategy_df is not None
            and not strategy_df.empty
            and strategy_diagnostics is not None
            and strategy_diagnostics["status"] != "data_unavailable"
        ):
            formatted = format_strategy_table(strategy_df.head(strategy_row_count))

            text_cols = ["Entry Rule", "Stop Rule", "Take Profit Rule", "11 AM Check", "Day Trader Plan"]
            main_cols = [c for c in formatted.columns if c not in text_cols]

            st.dataframe(
                formatted[main_cols],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Idea-level daily strategy suggestions for a roughly one-month expiry. "
                "Rows below the confidence threshold are filtered out before display. Confidence lower than 0.7 may not be recommended."
            )

            available_text_cols = [c for c in text_cols if c in formatted.columns]
            if available_text_cols and "Ticker" in formatted.columns:
                rules_source = formatted[["Ticker"] + [c for c in ["Horizon", "Expiration"] if c in formatted.columns] + available_text_cols].copy()

                rule_labels = rules_source["Ticker"].astype(str)
                if "Horizon" in rules_source.columns:
                    rule_labels = rule_labels + " | " + rules_source["Horizon"].astype(str)
                elif "Expiration" in rules_source.columns:
                    rule_labels = rule_labels + " | " + rules_source["Expiration"].astype(str)

                duplicate_counts = {}
                unique_labels = []
                for label in rule_labels:
                    count = duplicate_counts.get(label, 0) + 1
                    duplicate_counts[label] = count
                    unique_labels.append(f"{label} #{count}" if count > 1 else label)

                rules_source["Rule Label"] = unique_labels
                rules_df = (
                    rules_source[["Rule Label"] + available_text_cols]
                    .set_index("Rule Label")
                    .T
                )
                rules_df.index.name = "Rule"
                rules_df = rules_df.loc[:, ~rules_df.columns.duplicated()].copy()
                rules_df = rules_df.fillna("").map(str)
                rules_display_df = rules_df.reset_index()
                st.subheader("Trade Rules")
                st.dataframe(rules_display_df, use_container_width=True, hide_index=True)

        elif strategy_df is None:
            st.info("Click Run Strategy to generate fresh trade ideas.")

        st.subheader("Best Times to Run Strategy")
        best_times = pd.DataFrame([
            {
                "Time (ET)":       "9:45 – 10:15 AM",
                "Day":             "Tue – Thu",
                "Horizon Focus":   "Next Fri / Fri+2 (others) · All SPY horizons",
                "Why":             "Open noise settled, real volume confirming direction, tightest spreads",
                "Avoid If":        "Major news pre-announced (Fed, CPI, earnings on your tickers)",
            },
            {
                "Time (ET)":       "11:00 – 11:30 AM",
                "Day":             "Tue – Thu",
                "Horizon Focus":   "Next Fri / Fri+2 (others) · SPY +5d / +10d",
                "Why":             "Trend vs fade is clear, volume confirms or dies — the built-in 11 AM check",
                "Avoid If":        "SPY is flat ±0.2% — weak regime produces weak signals",
            },
            {
                "Time (ET)":       "Monday 10:00 – 11:00 AM",
                "Day":             "Monday only",
                "Horizon Focus":   "SPY +5d / +10d / +15d / +3wk for weekly bias",
                "Why":             "Gap from weekend resolved, sets tone for the week ahead",
                "Avoid If":        "Monday follows a 3-day weekend — extra volatility at open",
            },
            {
                "Time (ET)":       "Avoid: 9:30 – 9:40 AM",
                "Day":             "Any",
                "Horizon Focus":   "—",
                "Why":             "First 10 min erratic; spreads wide, volume spikes are noise",
                "Avoid If":        "Always skip this window",
            },
            {
                "Time (ET)":       "Avoid: Friday after 2 PM",
                "Day":             "Friday",
                "Horizon Focus":   "—",
                "Why":             "Premium sellers crush IV into weekend; Next Fri contracts lose value fast",
                "Avoid If":        "Always skip — bad pricing for Fri+2 as well",
            },
            {
                "Time (ET)":       "Avoid: Weekends / Pre-market",
                "Day":             "Sat – Sun",
                "Horizon Focus":   "—",
                "Why":             "yfinance returns stale/zero volume data; screener shows 'data unavailable'",
                "Avoid If":        "Always skip",
            },
        ])
        st.dataframe(best_times, use_container_width=True, hide_index=True)
        st.caption(
            "SPY rows give broad market context across 4 time horizons. "
            "Other tickers use the two nearest Fridays for near-term trade planning."
        )


    # =========================
    # OPTIONS CHAIN EXPLORER
    # =========================

    with tab5:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_options_chain = st.button("Run Options Chain Explorer", use_container_width=True)
        with col_info:
            if st.session_state["options_chain_fetched_at"]:
                st.caption(f"Last run: {st.session_state['options_chain_fetched_at']}")

        col1, col2 = st.columns(2)
        with col1:
            ticker = st.selectbox("Ticker", TICKERS)
        with col2:
            with st.spinner("Loading expirations..."):
                expirations = get_expirations_cached(ticker)
            if expirations:
                expiry = st.selectbox("Expiration", expirations)
            else:
                st.warning(f"No expirations available for {ticker}.")
                expiry = None

        if run_options_chain and expiry:
            with st.spinner(f"Fetching options chain for {ticker} {expiry}..."):
                try:
                    chain = get_options_chain(ticker, expiry)
                    flow = detect_unusual_flow(chain)
                    gex = calculate_gex(chain)

                    st.session_state["options_chain_df"] = format_options_table(chain)
                    st.session_state["options_flow_df"] = format_options_table(flow)
                    st.session_state["options_gex_series"] = gex
                    st.session_state["options_chain_fetched_at"] = _now()
                except Exception as e:
                    st.error(f"Failed to load options chain: {e}")

        if st.session_state["options_chain_df"] is not None:
            st.dataframe(
                st.session_state["options_chain_df"],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Unusual Options Flow")
            flow_df = st.session_state["options_flow_df"]
            if flow_df is not None and not flow_df.empty:
                st.dataframe(flow_df, use_container_width=True, hide_index=True)
            else:
                st.info("No unusual flow found for the latest run.")

            st.subheader("Dealer Gamma Exposure")
            gex = st.session_state["options_gex_series"]
            if gex is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=gex.index, y=gex.values))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a ticker and expiration, then click Run.")


    # =========================
    # PRICE FORECAST
    # =========================

    with tab6:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_forecast = st.button("Run Price Forecast", use_container_width=True)
        with col_info:
            if st.session_state["forecast_fetched_at"]:
                st.caption(f"Last run: {st.session_state['forecast_fetched_at']}")

        if run_forecast:
            cached_g = cm.load("forecast_gainers")
            cached_l = cm.load("forecast_losers")
            if cached_g and cached_l:
                st.session_state["forecast_gainers_df"] = pd.DataFrame(cached_g)
                st.session_state["forecast_losers_df"] = pd.DataFrame(cached_l)
                st.session_state["forecast_fetched_at"] = f"cached · {cm.age_label('forecast_gainers')}"
            else:
                _prog = st.progress(0, text=f"Bulk downloading price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
                try:
                    import data_fetcher as _df
                    _df.bulk_preload(MARKET_SCAN_TICKERS, period="1y")
                    _prog.progress(40, text="Price data ready — computing ML forecasts…")
                    gainers, losers = build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)
                    _prog.progress(100, text="Done!")
                    _prog.empty()
                    st.session_state["forecast_gainers_df"] = gainers
                    st.session_state["forecast_losers_df"] = losers
                    st.session_state["forecast_fetched_at"] = _now()
                    if gainers is not None and not gainers.empty:
                        cm.save("forecast_gainers", gainers.to_dict("records"))
                    if losers is not None and not losers.empty:
                        cm.save("forecast_losers", losers.to_dict("records"))
                except Exception as e:
                    _prog.empty()
                    st.warning(f"Forecast encountered an error: {e}. Showing any partial results.")

        gainers_df = st.session_state["forecast_gainers_df"]
        losers_df = st.session_state["forecast_losers_df"]

        _forecast_col_rename = {
            "ticker": "Ticker",
            "close": "Last Price",
            "one_week_view": "1W Direction",
            "est_1w_pct": "Est. 1W Move %",
            "est_2w_pct": "Est. 2W Move %",
            "forecast_confidence": "Confidence %",
            "rsi_14": "RSI (14)",
            "adx_14": "ADX (14)",
            "volatility_5d": "5D Range %",
            "volume_ratio_5": "Vol Ratio",
            "analyst_recom": "Analyst (1-5)",
            "target_upside_pct": "Target Upside %",
        }

        if gainers_df is not None and not gainers_df.empty:
            st.subheader("Top 10 Potential Gainers")
            st.dataframe(
                gainers_df.rename(columns=_forecast_col_rename),
                use_container_width=True,
                hide_index=True,
            )

        if losers_df is not None and not losers_df.empty:
            st.subheader("Top 10 Potential Losers")
            st.dataframe(
                losers_df.rename(columns=_forecast_col_rename),
                use_container_width=True,
                hide_index=True,
            )

        if gainers_df is None and losers_df is None:
            st.info("Click Run Price Forecast to scan tickers.")

        st.caption(
            "Estimates are derived from technical momentum signals (trend, RSI, MACD, ADX, MA alignment). "
            "They are not a prediction model — treat as a directional watchlist, not a guarantee. "
            "Est. 1W = next ~5 trading days. Est. 2W = next ~10 trading days."
        )


    # =========================
    # SHORT SQUEEZE SCANNER
    # =========================

    with tab7:

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_squeeze = st.button("Run Short Squeeze Scanner", use_container_width=True)
        with col_info:
            if st.session_state["squeeze_fetched_at"]:
                st.caption(f"Last run: {st.session_state['squeeze_fetched_at']}")

        col_sf, col_rows = st.columns(2)
        with col_sf:
            min_short_float = st.slider(
                "Min Short Float %", min_value=1, max_value=25, value=3,
                help="Only show stocks where shorts hold at least this % of the float",
            )
        with col_rows:
            squeeze_rows = st.slider("Max Results", min_value=5, max_value=30, value=15)

        st.caption(
            "Scans the full market universe for large-cap stocks (>$5B market cap, price >$10) "
            "with elevated short interest, rising momentum, and volume confirmation. "
            "Uses free yfinance data — short interest figures update twice a month."
        )

        if run_squeeze:
            st.session_state["squeeze_df"] = None
            _prog = st.progress(0, text=f"Bulk downloading 3-month price data for {len(MARKET_SCAN_TICKERS)} tickers… (~3-5s)")
            try:
                import data_fetcher as _df
                _df.bulk_preload(MARKET_SCAN_TICKERS, period="3mo")
                _prog.progress(40, text="Price data ready — fetching short interest & scoring squeeze candidates…")
                squeeze_df = scan_short_squeeze(
                    MARKET_SCAN_TICKERS,
                    min_short_float=min_short_float,
                )
                _prog.progress(100, text="Done!")
                _prog.empty()
                st.session_state["squeeze_df"] = squeeze_df
                st.session_state["squeeze_fetched_at"] = _now()
            except Exception as e:
                _prog.empty()
                st.error(f"Squeeze scan failed: {e}")

        squeeze_df = st.session_state["squeeze_df"]

        if squeeze_df is not None and not squeeze_df.empty:
            display = squeeze_df.head(squeeze_rows).copy()

            main_cols = {
                "ticker": "Ticker",
                "name": "Company",
                "sector": "Sector",
                "price": "Price",
                "market_cap_b": "Mkt Cap ($B)",
                "short_float_pct": "Short Float %",
                "days_to_cover": "Days to Cover",
                "volume_ratio": "Vol Ratio (5d/20d)",
                "ret_3d": "3D Return %",
                "ret_5d": "5D Return %",
                "rsi": "RSI (14)",
                "pct_above_ma20": "% vs 20MA",
                "squeeze_score": "Squeeze Score",
                "setup_quality": "Setup Tier",
            }
            available_main = {k: v for k, v in main_cols.items() if k in display.columns}
            st.subheader("Ranked Short Squeeze Candidates")
            st.dataframe(
                display[list(available_main.keys())].rename(columns=available_main),
                use_container_width=True,
                hide_index=True,
            )

            fig = px.bar(
                display,
                x="ticker",
                y="squeeze_score",
                color="setup_quality",
                text="squeeze_score",
                title="Squeeze Score by Ticker",
                labels={"ticker": "Ticker", "squeeze_score": "Squeeze Score", "setup_quality": "Tier"},
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 105], showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Why Each Ticker & When to Buy")
            for _, row in display.iterrows():
                tier_emoji = {"Tier 1": "🔴", "Tier 2": "🟠", "Tier 3": "🟡"}.get(
                    row["setup_quality"].split(" — ")[0].strip(), "⚪"
                )
                with st.expander(
                    f"{tier_emoji} {row['ticker']} — {row['setup_quality']}  |  "
                    f"Score: {row['squeeze_score']:.0f}  |  "
                    f"Short Float: {row['short_float_pct']:.1f}%  |  "
                    f"DTC: {row['days_to_cover']:.1f}d  |  "
                    f"Price: ${row['price']:.2f}"
                ):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Short Float %", f"{row['short_float_pct']:.1f}%")
                    col_b.metric("Days to Cover", f"{row['days_to_cover']:.1f}d")
                    col_c.metric("Volume Ratio", f"{row['volume_ratio']:.2f}x")

                    col_d, col_e, col_f = st.columns(3)
                    col_d.metric("3D Return", f"{row['ret_3d']:+.1f}%")
                    col_e.metric("RSI (14)", f"{row['rsi']:.0f}")
                    col_f.metric("vs 20-day MA", f"{row['pct_above_ma20']:+.1f}%")

                    st.markdown("**Why this ticker is a squeeze candidate:**")
                    for reason in row["reasons"].split(" | "):
                        st.markdown(f"- {reason}")

                    st.markdown("**Best time to buy:**")
                    st.info(row["buy_timing"])

                    if row.get("next_earnings") and row["next_earnings"] not in ("None", "nan", ""):
                        st.markdown(f"**Next earnings:** {row['next_earnings']}  _(catalyst window)_")

            st.caption(
                "Squeeze Score = weighted composite of short float %, days-to-cover, "
                "3d/5d momentum, volume ratio, and MA position. "
                "Short interest data from yfinance updates twice per month (FINRA settlement). "
                "Always confirm with a real-time source (Finviz, Ortex) before trading."
            )

        elif squeeze_df is not None:
            st.warning(f"No tickers met the minimum short float threshold of {min_short_float}%. Try lowering the filter.")
        else:
            st.info("Click **Run Short Squeeze Scanner** to find squeeze candidates.")


    # =========================
    # INTRADAY TIMING
    # =========================

    with tab8:

        st.markdown(
            "Estimates optimal **buy and sell windows** for each ticker based on 60 days of "
            "5-minute historical data. Shows average slot returns, volume profile, gap behaviour, "
            "and a plain-English recommendation per ticker. Data is cached for 4 hours."
        )

        col_run, col_info = st.columns([1, 4])
        with col_run:
            run_timing = st.button("Run Intraday Timing", use_container_width=True)
        with col_info:
            if st.session_state["timing_fetched_at"]:
                st.caption(f"Last run: {st.session_state['timing_fetched_at']}")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            timing_tickers_input = st.text_input(
                "Tickers (comma-separated, or leave blank to use Market Universe)",
                value="",
                placeholder="e.g. AAPL, NVDA, TSLA",
            )
        with col_t2:
            timing_contract = st.selectbox("Direction", ["call", "put", "both"], index=0)

        if run_timing:
            raw_input = timing_tickers_input.strip()
            if raw_input:
                timing_ticker_list = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
            else:
                timing_ticker_list = MARKET_SCAN_TICKERS

            if timing_contract == "both":
                pairs = [(t, "call") for t in timing_ticker_list] + [(t, "put") for t in timing_ticker_list]
            else:
                pairs = [(t, timing_contract) for t in timing_ticker_list]

            st.session_state["timing_results"] = None
            with st.spinner(f"Analysing {len(timing_ticker_list)} tickers × {'2 directions' if timing_contract == 'both' else '1 direction'}…"):
                results = get_intraday_timing_batch(pairs, workers=6)
                st.session_state["timing_results"] = results
                st.session_state["timing_fetched_at"] = _now()

        timing_results = st.session_state["timing_results"]

        if timing_results:
            summary_rows = []
            for (ticker, ct), data in sorted(timing_results.items()):
                if data.get("error"):
                    continue
                summary_rows.append({
                    "Ticker": ticker,
                    "Direction": ct.upper(),
                    "Best Entry": data["best_entry_window"],
                    "Best Exit": data["best_exit_window"],
                    "Gap Fade %": f"{data['gap_fade_prob']:.0f}%" if data["gap_fade_prob"] is not None else "—",
                    "Days Analysed": data["n_days"],
                })

            if summary_rows:
                st.subheader("Summary — Optimal Windows")
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Ticker Detail — Volume Profile & Slot Returns")

            seen = set()
            for (ticker, ct), data in sorted(timing_results.items()):
                card_key = f"{ticker}_{ct}"
                if card_key in seen:
                    continue
                seen.add(card_key)

                if data.get("error"):
                    with st.expander(f"{ticker} ({ct.upper()}) — ⚠️ {data['error']}"):
                        st.warning(data["error"])
                    continue

                entry_w = data["best_entry_window"]
                exit_w = data["best_exit_window"]
                n_days = data["n_days"]

                with st.expander(
                    f"{ticker} ({ct.upper()}) — Entry: {entry_w}  |  Exit: {exit_w}  |  {n_days}d history"
                ):
                    st.markdown("**Recommendation**")
                    st.info(data["timing_note"])

                    slot_df = pd.DataFrame({
                        "Time Slot": TIMING_SLOT_LABELS,
                        "Avg Return %": data["avg_return_by_slot"],
                        "Win Rate %": data["win_rate_by_slot"],
                        "Vol vs Avg": data["volume_profile"],
                    })
                    slot_df["Avg Return %"] = slot_df["Avg Return %"].round(3)
                    slot_df["Win Rate %"] = slot_df["Win Rate %"].round(1)
                    slot_df["Vol vs Avg"] = slot_df["Vol vs Avg"].round(2)

                    entry_label = data["best_entry_window"]
                    exit_label = data["best_exit_window"]

                    def _highlight_slots(row):
                        if row["Time Slot"] == entry_label:
                            return ["background-color: #1a472a; color: white"] * len(row)
                        if row["Time Slot"] == exit_label:
                            return ["background-color: #1a3a5c; color: white"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        slot_df.style.apply(_highlight_slots, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

                    fig_vol = px.bar(
                        slot_df,
                        x="Time Slot",
                        y="Vol vs Avg",
                        title=f"{ticker} — Volume Profile (vs daily avg)",
                        color="Vol vs Avg",
                        color_continuous_scale="Blues",
                    )
                    fig_vol.update_layout(
                        xaxis_tickangle=-45,
                        coloraxis_showscale=False,
                        height=300,
                        margin=dict(t=40, b=80),
                    )
                    fig_vol.add_hline(y=1.0, line_dash="dot", line_color="gray",
                                      annotation_text="Avg", annotation_position="top right")
                    st.plotly_chart(fig_vol, use_container_width=True)

                    fig_ret = px.bar(
                        slot_df,
                        x="Time Slot",
                        y="Avg Return %",
                        title=f"{ticker} ({ct.upper()}) — Avg Slot Return over {n_days} days",
                        color="Avg Return %",
                        color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0,
                    )
                    fig_ret.update_layout(
                        xaxis_tickangle=-45,
                        coloraxis_showscale=False,
                        height=300,
                        margin=dict(t=40, b=80),
                    )
                    fig_ret.add_hline(y=0, line_dash="dot", line_color="gray")
                    st.plotly_chart(fig_ret, use_container_width=True)

            st.caption(
                "Based on 60 days of 5-min yfinance data. Slot returns are averages — "
                "actual performance varies. Gap fade probability is computed from days where the "
                "opening gap was ≥0.5% vs prior close. Green row = suggested entry, blue = suggested exit."
            )

        else:
            st.info("Enter tickers (or leave blank for market universe) and click **Run Intraday Timing**.")


    # =========================
    # TICKER ANALYSIS
    # =========================

    with tab9:
        _render_ticker_analysis()
