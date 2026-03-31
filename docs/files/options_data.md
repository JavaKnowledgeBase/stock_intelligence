# File Reference: options_data.py

## Purpose

The core business logic engine for the entire platform. Every piece of data displayed in the Streamlit dashboard flows through this file. It is responsible for:

- Fetching and caching live price history and options chains from Yahoo Finance
- Computing market mover scores (multi-factor directional signals per ticker)
- Building volume leader rankings with historical trend tracking
- Generating strategy recommendations (contract selection, trade rules, scoring)
- Producing price forecast estimates (directional % move estimates)

---

## Libraries Used

| Library | Why |
|---|---|
| `yfinance` | Downloads OHLCV price history and options chain data from Yahoo Finance |
| `pandas` | All DataFrame operations: merge, groupby, pivot_table, rolling, shift |
| `numpy` | Numeric operations: np.isfinite, np.log, np.sqrt, np.clip |
| `threading` | `threading.Lock` protects the shared in-memory cache from concurrent writes |
| `concurrent.futures` | `ThreadPoolExecutor` for parallel per-ticker data fetching |
| `pathlib` | Path manipulation for the daily snapshot CSV directory |
| `time` | TTL expiry check for cache entries |
| `config` | Imports `DATA_DIR` for the snapshot directory path |
| `build_features` | Calls `build_features()` to compute technical indicators from raw OHLCV |

---

## Caching System

### Constants

```python
_CACHE_TTL_SECONDS = 900   # 15 minutes
_CACHE = {}                # {key: (timestamp, value)}
_CACHE_LOCK = threading.Lock()
_MAX_WORKERS = 8
```

### Cache Key Format

```python
_cache_key("history",      ticker, period, interval)  → ("history", "AAPL", "1y", "1d")
_cache_key("chain",        ticker, expiry)             → ("chain", "AAPL", "2026-04-17")
_cache_key("expirations",  ticker)                     → ("expirations", "AAPL")
_cache_key("ticker",       ticker)                     → ("ticker", "AAPL")
```

### Cache Functions

- `_get_cached(key)` — acquires lock, checks TTL, returns value or None
- `_set_cached(key, value)` — acquires lock, stores `(time.time(), value)`, returns value
- `_clone_frame(df)` — returns `df.copy()` for DataFrames, raw value otherwise (prevents caller mutation of cached data)

---

## Utility Functions

### `_flatten_df_columns(df)`
Handles yfinance MultiIndex columns across API versions. Iterates levels of the MultiIndex looking for the one containing recognised price column names (`{'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}`). Uses that level as flat columns. Fallback: takes `col[0]` from each tuple.

### `_safe_float(value, default, minimum, maximum)`
Coerces any value to float using `pd.to_numeric(errors="coerce")`, checks for NaN and infinity, falls back to `default`, then applies optional min/max bounds. Used everywhere yfinance-sourced numbers are consumed.

### `_clean_numeric_columns(df, defaults)`
Batch version of `_safe_float` for a DataFrame. Replaces inf/−inf with NaN, then fills NaN with per-column defaults.

### `_is_rate_limited_error(exc)`
Checks if an exception message contains "too many requests", "rate limited", or "429". Used throughout to detect Yahoo Finance throttling.

---

## Data Fetching Functions

### `get_price_history(ticker, period, interval)`
Uses `yf.Ticker(ticker).history()` — NOT `yf.download()`, which has a thread-safety bug causing response mixing under parallel load. Returns a OHLCV DataFrame. Cached by (ticker, period, interval).

### `get_stock_price(ticker)`
Returns the most recent Close from 5-day history as a float.

### `get_expirations(ticker)`
Returns a list of available expiration date strings for the ticker's options. Cached.

### `get_options_chain(ticker, expiry)`
Fetches calls and puts for a specific expiry, adds a `"type"` column ("call"/"put"), and concatenates them. Cached by (ticker, expiry).

### `get_market_options_snapshot(ticker, max_contracts)`
Returns the top `max_contracts` (default 10) most active option contracts for the nearest expiry. Sorted by volume, open_interest, last_price descending.

### `get_full_market_options_snapshot(ticker)`
Full chain (all strikes) for the nearest expiry with numeric coercion. Used by the volume table builder.

---

## Volume Table Builder

### `build_market_volume_table(tickers, lookback_days=7)`

**Purpose:** Produce a ranked table of tickers by today's options volume with 7-day historical trend.

**Inputs:** List of tickers, optional lookback window.

**Process:**
1. Parallel `_fetch_volume_row()` for each ticker — sums call and put volume from the full snapshot
2. Saves today's row to `data/options_market_snapshots/YYYY-MM-DD.csv`
3. Reads all CSVs in that directory into a history DataFrame
4. Computes 7-day totals and trend via `pivot_table` + half-period comparison
5. Merges daily + weekly + trend data; sorts by one_day_volume desc

**Returns:** DataFrame with columns: ticker, dominant_side, one_day_volume, percent_of_day_total, one_week_total, one_week_trend, call_volume, put_volume

**Persistent side effect:** Writes a new CSV file per trading day to `data/options_market_snapshots/`.

---

## Market Mover Builder

### `_build_mover_row_inner(ticker)` → dict | None

**Purpose:** Score a single ticker for 1-week and 1-month directional potential.

**Returns None if:** history fetch fails, fewer than 40 rows, any required metric is NaN.

**Key metrics extracted from build_features() output:**

| Feature | Variable | Used As |
|---|---|---|
| close_ret_5d | ret_5d | Trend direction signal |
| close_ret_3d | ret_3d | Short-term trend |
| volatility_5d | vol_5d | Swing amplitude |
| volatility_10d | vol_10d | Swing amplitude |
| volume_ratio_5 | volume_ratio_5 | Volume confirmation |
| volume_ratio_20 | volume_ratio_20 | Volume confirmation |
| dist_ma_20_pct | dist_ma_20 | Extension from mean |
| dist_ema_20_pct | dist_ema_20 | Extension from mean |
| rsi_14 | rsi_14 | Momentum zone |
| macd_hist | macd_hist | MACD confirmation |
| adx_14 | adx_14 | Trend strength |
| ma_alignment | ma_alignment | MA stack bias |
| trend_consistency_10d | trend_consistency | Day-over-day momentum |
| vol_direction_ratio | vol_dir_ratio | Up-day vs down-day volume |
| obv_slope_10d | obv_slope | OBV accumulation |
| pct_from_52w_high | pct_from_52w_high | Breakout proximity |
| atr_14 | atr_14 | ATR for position sizing |

**Scoring (upside/downside separately):**

```
trend_factor    = ret_5d × 0.65 + ret_3d × 0.35
swing_factor    = vol_5d × 0.6  + vol_10d × 0.4
volume_factor   = vol_ratio_5 × 0.6 + vol_ratio_20 × 0.4
extension_factor= dist_ma_20 × 0.6  + dist_ema_20 × 0.4

1W_upside  = trend×0.9  + swing×0.8  + max(vol−1,0)×6  + max(ext,0)×0.35
           + RSI_adj_up + MACD_adj + ADX_adj + MA_adj_up + TC_adj_up
           + VD_adj_up + OBV_adj_up + 52W_adj_up + RS_adj_up

1W_downside = (−trend)×0.9 + swing×0.8 + max(vol−1,0)×6 + max(−ext,0)×0.35
            + RSI_adj_dn + (−MACD_adj) + ADX_adj + MA_adj_dn + ...
```

**Additive adjustments:**

| Factor | Upside Boost | Condition |
|---|---|---|
| RSI | +2.0 | 50 ≤ RSI ≤ 70 |
| RSI | −3.0 | RSI > 75 |
| MACD | +1.5 | macd_hist > 0 |
| ADX | −2.0 | ADX < 20 (weak trend) |
| ADX | +2.5 | ADX > 40 (strong trend) |
| MA stack | +2.0 | Bullish alignment |
| Trend consistency | +1.5 | >60% positive days |
| Vol direction | +1.0 | up_vol > 1.2× down_vol |
| OBV slope | +0.75 | slope > 5 |
| 52W proximity | +1.0 | within 3% of high |
| Rel strength vs SPY | +1.0 | outperforming by >2% |

**Returns:** dict with ticker, close, one_week_view, one_week_score, one_month_view, one_month_score, ret_5d, volatility_5d, volume_ratio_5, rsi_14, adx_14, atr_14, rel_strength_vs_spy, pct_from_52w_high, obv_slope_10d, hv_30d

### `build_market_movers_table(tickers)` → DataFrame
Parallel `_build_mover_row` across all tickers. Sorted by one_week_score descending.

---

## Price Forecast Builder

### `build_price_forecast_table(tickers, top_n=10)` → (gainers_df, losers_df)

Uses `build_market_movers_table()` output. Computes:

```
forecast_confidence = clip(ADX/35, 0.65, 1.15)
                    × clip(vol_ratio_5/1.5, 0.75, 1.15)
                    × clip(1 − |RSI−50|/100, 0.70, 1.00)

vol_cap   = (volatility_5d × 5).clip(lower=2.0)

est_1w_pct = min(1W_score × 0.30 × confidence, vol_cap)
est_2w_pct = min(2W_score × 0.40 × confidence, vol_cap × 1.25)
```

Direction applied via mask: gainers get positive sign, losers get negative.

Returns top_n rows each sorted by |est_1w_pct|.

---

## Strategy Contract Picker

### `_pick_strategy_contract(ticker, contract_type, close_price, hv_30d, expiry)` → dict | None

**Purpose:** Find the single best-quality tradeable option contract for a given ticker, direction, and expiry.

**Process:**
1. Fetch full options chain for the expiry
2. Filter to `contract_type` (call/put)
3. Coerce all numeric columns; compute `mid_price` and `spread_pct`
4. `base_filter`: ask > 0, bid ≥ 0, mid > 0
5. **Strike range filter** (prevents deep ITM accumulation bias):
   - Calls: `[close × 0.95, close × 1.15]`
   - Puts: `[close × 0.85, close × 1.05]`
6. Two-pass liquidity filter:
   - **Tight:** volume ≥ 100, OI ≥ 100, spread ≤ 12%
   - **Loose fallback:** volume ≥ 50, OI ≥ 100, spread ≤ 18%
7. Compute `contract_quality_score`:
   ```
   target_strike = close × 1.02 (call) or close × 0.98 (put)
   quality = (liq/max_liq × 55) + ((18−spread).clip(0)/18 × 30) + ((10−dist%).clip(0)/10 × 15)
   ```
8. Select `working.iloc[0]` (highest quality score)
9. Compute IV/HV ratio if `impliedVolatility` is available

**Returns:** strike, option_value (mid_price), bid, ask, spread_pct, open_interest, volume, contract_quality_score, iv_hv_ratio, tight_liquidity (bool)

---

## Strategy Recommendation Builder

### `_build_strategy_recommendation(row)` → dict | None

**Purpose:** Build a complete trade recommendation from a scored candidate row.

**ATR-based trade sizing:**
```
atr_pct    = atr_14 / close_price
stop_mult  = clip(atr_pct × 1.5, 0.15, 0.30)
t1_mult    = clip(atr_pct × 2.0, 0.20, 0.35)
t2_mult    = clip(atr_pct × 3.0, 0.30, 0.50)

stop_price = mid_price × (1 − stop_mult)
target_1   = mid_price × (1 + t1_mult)
target_2   = mid_price × (1 + t2_mult)
```

**Text rules generated:** entry_rule, stop_rule, take_profit_rule, midday_check, daily_plan — all include concrete dollar values.

---

## Strategy Table Builder

### `build_strategy_table(tickers, top_n=10)` → (DataFrame, diagnostics_dict)

**Orchestrates the full pipeline:**
1. `build_market_movers_table()` + `build_market_volume_table()` (parallel, error-isolated)
2. Merge on ticker; clean numerics; filter for valid rows
3. Compute `strategy_confidence` (same formula as forecast)
4. Filter confidence ≥ 0.60
5. Compute `strategy_score = (1W×0.35 + 1M×0.45 + vol%×0.20) × confidence`
6. Detect market regime; apply regime filter (suppress weak counter-trend signals)
7. Expand top_n tickers × 2 horizons (Next Fri, Fri+2) — SPY gets 4 horizons
8. Parallel `_build_strategy_recommendation` for all expanded rows
9. Sort: SPY first, then by strategy_score desc

**Diagnostics dict keys:**
- `tickers_requested`, `movers_found`, `volume_rows_found`
- `combined_candidates`, `contracts_evaluated`, `contracts_selected`
- `status`: "ok" | "data_unavailable" | "no_contracts" | "partial_results"
- `message`, `rate_limited`, `expiration_errors`, `contract_errors`, `regime`

---

## Helper Functions for Expiry Selection

### `_get_strategy_expiration(ticker, min_days=21, max_days=45)`
Prefers expirations 21–45 days out (theta-friendly range). Falls back to closest expiry at least 7 days out if no preferred exists.

### `_get_expiration_near_date(ticker, target_date)`
Returns the available expiry closest in days to `target_date`. Used to match "Next Fri" / "+5d" horizon labels to real expiry dates.

### `_upcoming_fridays(n=2)`
Returns the next n Fridays from today as Timestamps.

### `_get_market_regime()`
Returns "bullish", "bearish", or "neutral" based on SPY's 5-day return and 20-day MA position.

### `_get_spy_ret_5d()`
Cached SPY 5-day return as a float. Used for relative strength calculation in `_build_mover_row_inner`.
