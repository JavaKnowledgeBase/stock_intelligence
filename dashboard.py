# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 06:54:23 2026

@author: rkafl
"""

import concurrent.futures

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MARKET_SCAN_TICKERS, TICKERS
from gex_engine import calculate_gex
from options_data import (
    build_market_movers_table,
    build_strategy_table,
    build_market_volume_table,
    get_expirations,
    get_market_options_snapshot,
    get_options_chain,
)
from options_flow import detect_unusual_flow


st.set_page_config(layout="wide")

st.title("Institutional Options Intelligence Platform")


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

    formatted_df = df[[
        "ticker", "view", "contract_type", "expiration", "strike_price",
        "option_value", "bid", "ask", "spread_pct", "open_interest", "option_volume",
        "contract_quality_score", "strategy_score", "day_volume_share",
        "underlying_5d_move", "entry_rule", "stop_rule", "take_profit_rule",
        "midday_check", "daily_plan",
    ]].copy()

    return formatted_df.rename(columns={
        "ticker": "Ticker",
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
        "day_volume_share": "% of Day Volume",
        "underlying_5d_move": "Underlying 5D Move %",
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
]:
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Options Screener",
    "Market Volume Leaders",
    "Rapid Movers",
    "Strategy Ideas",
    "Options Chain Explorer",
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
                st.error(f"Strategy run failed: {e}")

    strategy_df = st.session_state["strategy_df"]
    strategy_diagnostics = st.session_state["strategy_diagnostics"]

    if strategy_diagnostics is not None:
        freshness_label = (
            "Fresh data" if strategy_diagnostics["status"] != "data_unavailable"
            else "Stale data blocked"
        )
        st.caption(
            f"{freshness_label} | Status: {strategy_diagnostics['status']} | "
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
        st.dataframe(
            format_strategy_table(strategy_df.head(strategy_row_count)),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Idea-level daily strategy suggestions for a roughly one-month expiry. "
            "Intended as a watchlist, not a guarantee."
        )
    elif strategy_df is None:
        st.info("Click Run Strategy to generate fresh trade ideas.")


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
