# File Reference: dashboard.py

## Purpose

The main entry point for the Streamlit web application. Defines the full six-tab UI layout, wires user interactions to the backend data functions in `options_data.py`, and manages all session state. This file contains no business logic — it is purely a presentation and orchestration layer.

---

## Libraries Used

| Library | Why |
|---|---|
| `streamlit` | Web framework: tabs, buttons, sliders, dataframes, charts, session state, caching |
| `plotly.express` | Scatter plot for the options screener tab |
| `plotly.graph_objects` | Bar chart for the GEX visualisation |
| `pandas` | DataFrame handling for filter operations and display formatting |
| `concurrent.futures` | `ThreadPoolExecutor` for parallel snapshot fetching |
| `os` | Read environment variables for build label |
| `subprocess` | Run `git rev-parse --short HEAD` to get build commit hash |

---

## Startup

```python
ensure_assets_available()    # Sync models/data from R2 if needed
st.set_page_config(layout="wide")
```

`ensure_assets_available()` runs before any UI is rendered — on a fresh deployment this downloads trained models from R2.

---

## Build Label

```python
@st.cache_data(show_spinner=False)
def get_build_label():
```

Reads `STREAMLIT_BUILD_COMMIT`, `GITHUB_SHA`, or `COMMIT_SHA` environment variables. Falls back to running `git rev-parse --short HEAD`. Displayed in the subtitle as `Branch: main | Build: f1d65c2`.

---

## Session State

Session state variables persist across Streamlit reruns within the same browser session. All are initialised at startup:

| Key | Type | Description |
|---|---|---|
| `market_options_raw_df` | DataFrame or None | Unfiltered screener results (filters reapply without re-fetch) |
| `market_options_failed` | list | Tickers that failed or timed out |
| `market_options_fetched_at` | str or None | Timestamp of last run |
| `market_volume_df` | DataFrame or None | Volume leaders table |
| `market_volume_fetched_at` | str or None | Timestamp |
| `market_movers_df` | DataFrame or None | Rapid movers table |
| `market_movers_fetched_at` | str or None | Timestamp |
| `strategy_df` | DataFrame or None | Strategy ideas table |
| `strategy_diagnostics` | dict or None | Diagnostics from build_strategy_table |
| `strategy_fetched_at` | str or None | Timestamp |
| `options_chain_df` | DataFrame or None | Formatted options chain |
| `options_flow_df` | DataFrame or None | Unusual flow subset |
| `options_gex_series` | Series or None | GEX by strike |
| `options_chain_fetched_at` | str or None | Timestamp |
| `forecast_gainers_df` | DataFrame or None | Top 10 predicted gainers |
| `forecast_losers_df` | DataFrame or None | Top 10 predicted losers |
| `forecast_fetched_at` | str or None | Timestamp |

---

## Parallel Snapshot Fetcher

### `_fetch_snapshot(ticker, max_contracts)` → (ticker, DataFrame, error)
Calls `get_market_options_snapshot()` and wraps in try/except to return errors without raising.

### `fetch_all_snapshots(tickers, max_contracts, timeout=45, workers=10)` → (snapshots, failed)
Uses `ThreadPoolExecutor` with `concurrent.futures.wait(timeout=45)`. Any futures still pending after 45 seconds are cancelled. Failed tickers are collected separately. Returns non-empty DataFrames concatenated.

---

## Table Formatters

All formatters are pure functions: select a subset of columns, rename them for display, return a copy.

### `format_options_table(df)`
Selects: type, strike, lastPrice, bid, ask, volume, openInterest, inTheMoney, impliedVolatility, lastTradeDate.
Renames to: Call or Put, Strike, Last Price, Bid, Ask, Volume, Open Interest, In The Money, Implied Volatility, Last Trade Date.

### `format_market_volume_table(df)`
Selects: ticker, dominant_side, one_day_volume, percent_of_day_total, one_week_total, one_week_trend, call_volume, put_volume.
Rounds `percent_of_day_total` to 2 decimal places.

### `format_market_movers_table(df)`
Selects: ticker, close, one_week_view, one_week_score, one_month_view, one_month_score, ret_5d, volatility_5d, volume_ratio_5.

### `format_strategy_table(df)`
Selects base columns + optional (rsi_14, adx_14, iv_hv_ratio if present) + text rule columns.
Returns renamed DataFrame with all columns in display-friendly names.

---

## Tab 1 — Market Options Screener

**Controls:** contract_type_filter, itm_filter, min_volume (number_input), contracts_per_ticker (slider 3–20)

**On run:** calls `fetch_all_snapshots(TICKERS, contracts_per_ticker, timeout=45, workers=10)`. Stores raw result in session state.

**Filtering:** Applied live on `raw_df` without re-fetching — this is the key UX pattern for this tab. Filters applied in sequence: volume, contract type, ITM status. Sorted by volume, OI, last_price descending.

**Display:**
- Row count caption (ticker count from raw_df)
- `st.dataframe()` of renamed columns
- `px.scatter()` with x=strike, y=last_price, size=volume, color=option_type, hover data showing expiration/bid/ask/OI

---

## Tab 2 — Market Volume Leaders

**Controls:** `volume_leaders_count` slider (10–50, default 20)

**On run:** calls `build_market_volume_table(MARKET_SCAN_TICKERS)`. Stores in session state.

**Display:** `format_market_volume_table()` applied to head(volume_leaders_count). Caption explains the free-data approximation and weekly totals methodology.

---

## Tab 3 — Rapid Movers

**Controls:** `movers_row_count` slider (10–20, default 10)

**On run:** calls `build_market_movers_table(MARKET_SCAN_TICKERS)`. Stores in session state.

**Display:** `format_market_movers_table()` applied to head(movers_row_count). Caption explains the free-data nature of the signals.

---

## Tab 4 — Strategy Ideas

**Controls:** `strategy_row_count` slider (5–15, default 10)

**On run:** clears previous `strategy_df` and `strategy_diagnostics` from session state, then calls `build_strategy_table(MARKET_SCAN_TICKERS, top_n=strategy_row_count)`.

**Diagnostics banner:** Always shown if diagnostics exist. Displays: freshness_label, status, market regime, ticker/mover/volume/candidate/evaluated/selected counts. Shows warning/info based on status.

**Main table:** `format_strategy_table()` applied, text columns excluded from main view (shown separately below).

**Trade Rules pivot:** The `["Entry Rule", "Stop Rule", "Take Profit Rule", "11 AM Check", "Day Trader Plan"]` columns are transposed so rules are rows and tickers/horizons are columns.

**Duplicate column handling:** Since the same ticker can appear for multiple horizons, rule column labels are `f"{ticker} | {horizon}"` — a deduplication loop appends `#2`, `#3` etc. if the same label appears more than once.

**Best Times table:** Static DataFrame hardcoded in the file, always displayed below the results.

---

## Tab 5 — Options Chain Explorer

**Controls:** ticker selectbox (from TICKERS), expiry selectbox (from `get_expirations_cached(ticker)`)

`get_expirations_cached` uses `@st.cache_data(ttl=3600)` — expirations refresh hourly, not per-rerun.

**On run:** calls `get_options_chain()`, `detect_unusual_flow()`, `calculate_gex()`. Stores all three in session state.

**Display:**
- Full chain as `st.dataframe()`
- Unusual flow as `st.dataframe()` or "No unusual flow found" info box
- GEX as `go.Figure()` bar chart with `fig.add_trace(go.Bar(x=gex.index, y=gex.values))`

---

## Tab 6 — Price Forecast

**On run:** calls `build_price_forecast_table(MARKET_SCAN_TICKERS, top_n=10)`. Returns (gainers, losers) tuple.

**Display:** Two subheaders with separate `st.dataframe()` for gainers and losers, using the `_forecast_col_rename` dict inline (not a formatter function).

**Column rename dict** (`_forecast_col_rename`): defined at tab level for reuse on both gainers and losers DataFrames.

---

## Key Design Patterns

**Live filtering without re-fetching:** Tab 1 stores raw data in session state and reapplies filters on every rerun. This avoids yfinance calls for simple filter changes — the network call only happens when the user explicitly clicks "Run".

**Spinner + session state pattern:**
```python
if run_button:
    with st.spinner("..."):
        result = expensive_function()
    st.session_state["key"] = result

data = st.session_state["key"]
if data is not None:
    st.dataframe(data)
```

**State isolation on re-run:** Strategy tab clears `strategy_df` and `strategy_diagnostics` to None before the spinner — ensures stale results don't persist if the new run fails halfway.

**Timestamp helper:** `_now()` returns `pd.Timestamp.now().strftime("%Y-%m-%d %I:%M:%S %p")` — used for all "Last run:" captions.
