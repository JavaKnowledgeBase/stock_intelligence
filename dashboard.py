# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:54:23 2026

@author: rkafl
"""

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


ensure_assets_available()

st.set_page_config(layout="wide")


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


st.title("Institutional Options Intelligence Platform")
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
]:
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Market Options Screener",
    "Market Volume Leaders",
    "Rapid Movers",
    "Strategy Ideas",
    "Options Chain Explorer",
    "Price Forecast",
    "Short Squeeze Scanner",
])


# =========================
# MARKET SCREENER
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
        # Filters reapply live without re-running
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
        with st.spinner(f"Scanning {len(MARKET_SCAN_TICKERS)} tickers..."):
            try:
                result = build_market_volume_table(MARKET_SCAN_TICKERS)
                st.session_state["market_volume_df"] = result
                st.session_state["market_volume_fetched_at"] = _now()
            except Exception as e:
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
        with st.spinner(f"Scanning {len(MARKET_SCAN_TICKERS)} tickers for rapid move candidates..."):
            try:
                result = build_market_movers_table(MARKET_SCAN_TICKERS)
                st.session_state["market_movers_df"] = result
                st.session_state["market_movers_fetched_at"] = _now()
            except Exception as e:
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

        with st.spinner("Building strategy ideas from the latest market signals..."):
            try:
                strategy_df, strategy_diagnostics = build_strategy_table(
                    MARKET_SCAN_TICKERS,
                    top_n=strategy_row_count,
                )
                st.session_state["strategy_diagnostics"] = strategy_diagnostics
                st.session_state["strategy_df"] = (
                    pd.DataFrame() if strategy_diagnostics["status"] == "data_unavailable"
                    else strategy_df
                )
                st.session_state["strategy_fetched_at"] = _now()
            except Exception as e:
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

        # Trade rules: pivot so rules are rows and ideas are columns.
        # Streamlit's Arrow serializer requires unique column names.
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

    # Always visible — independent of results
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
        with st.spinner("Scanning tickers and computing forecasts..."):
            try:
                gainers, losers = build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)
                st.session_state["forecast_gainers_df"] = gainers
                st.session_state["forecast_losers_df"] = losers
                st.session_state["forecast_fetched_at"] = _now()
            except Exception as e:
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
            "Min Short Float %", min_value=5, max_value=25, value=8,
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
        with st.spinner(f"Scanning {len(MARKET_SCAN_TICKERS)} tickers for squeeze setups…"):
            try:
                squeeze_df = scan_short_squeeze(
                    MARKET_SCAN_TICKERS,
                    min_short_float=min_short_float,
                )
                st.session_state["squeeze_df"] = squeeze_df
                st.session_state["squeeze_fetched_at"] = _now()
            except Exception as e:
                st.error(f"Squeeze scan failed: {e}")

    squeeze_df = st.session_state["squeeze_df"]

    if squeeze_df is not None and not squeeze_df.empty:
        display = squeeze_df.head(squeeze_rows).copy()

        # Main metrics table
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

        # Score bar chart
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

        # Detailed candidate cards
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

