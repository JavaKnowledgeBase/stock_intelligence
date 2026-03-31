# File Reference: Supporting Python Files

## config.py

### Purpose
Central configuration file. Single source of truth for ticker lists and directory paths.

### Constants

**`TICKERS`** (30 tickers)
Core universe for the Market Options Screener tab. Large/mega-cap stocks with liquid options:
AAPL, AMD, AMZN, AVGO, BAC, C, CMCSA, CSCO, CVX, DIS, GOOG, HOOD, HPE, INTC, JPM, META, MSFT, MU, NFLX, NVDA, ORCL, PFE, PLTR, PSFT, QQQ, SPY, TSLA, V, VZ, XOM

**`MARKET_SCAN_TICKERS`** (70+ tickers)
Wider universe for the Strategy, Volume Leaders, Rapid Movers, and Forecast tabs. Includes all TICKERS plus sector ETFs and additional mid-large caps: XLK, XLE, XLF, XLV, XLI, XLY, COST, TGT, WMT, etc.

**`DATA_DIR`** = `"data/"` — base path for all local persistent data.

---

## gex_engine.py

### Purpose
Computes Dealer Gamma Exposure (GEX) by strike price from an options chain DataFrame.

### Libraries
`pandas`, `numpy`

### Function: `calculate_gex(chain_df)` → Series

```python
gex = chain_df["gamma"] * chain_df["openInterest"] * 100 * chain_df["strike"]
return gex.groupby(chain_df["strike"]).sum()
```

**Returns:** pandas Series indexed by strike price, values = net GEX at that strike.

**Interpretation:**
- Positive bars = dealers are net long gamma; they stabilise price (buy dips, sell rips)
- Negative bars = dealers are net short gamma; they amplify price moves
- The strike with the largest positive bar is often a price magnet for the day

**Used in:** dashboard.py Tab 5 as a `go.Bar` chart.

**Note:** `calculate_gex.py` in the root is an older duplicate of this file. `gex_engine.py` is the canonical version.

---

## options_flow.py

### Purpose
Detects unusual options activity — contracts where today's trading volume is anomalously high relative to existing open interest.

### Libraries
`pandas`, `numpy`

### Function: `detect_unusual_flow(chain_df)` → DataFrame

```python
return chain_df[
    (chain_df["volume"] > 500) &
    (chain_df["volume"] / chain_df["openInterest"].replace(0, np.nan) > 2)
].sort_values("volume", ascending=False)
```

**Thresholds:**
- Absolute: volume > 500 contracts
- Relative: volume/OI ratio > 2 (today's trading is more than twice the standing positions)

**Interpretation:** A contract meeting both conditions suggests new institutional money entering a position, rather than routine hedging or rolling. This is sometimes called a "sweep" or "block trade" signal.

**Used in:** dashboard.py Tab 5 as the "Unusual Options Flow" table.

**Note:** `detect_unusual_flow.py` in the root is a legacy duplicate. `options_flow.py` is canonical.

---

## volatility.py

### Purpose
Standalone utility functions for volatility analysis.

### Functions

**`calculate_historical_volatility(prices, window=20)`** → float
Annualised historical volatility: `std(log_returns) × sqrt(252) × 100`.

**`calculate_iv_rank(current_iv, iv_history)`** → float
IV Rank = `(current_iv - min_iv) / (max_iv - min_iv) × 100`. Measures where current IV sits within its 52-week range. 0 = at yearly low, 100 = at yearly high.

**`calculate_expected_move(price, iv, days_to_expiry)`** → float
Expected move = `price × iv × sqrt(days/365)`. The ±1 standard deviation range the market prices in for the expiry.

---

## strategy_engine.py

### Purpose
Simple rule-based strategy suggestion based on directional view and IV rank.

### Function: `suggest_strategy(direction, iv_rank)` → str

| Direction | IV Rank | Suggestion |
|---|---|---|
| bullish | > 50 | "Sell put spread (IV rich)" |
| bullish | ≤ 50 | "Buy call (IV cheap)" |
| bearish | > 50 | "Sell call spread (IV rich)" |
| bearish | ≤ 50 | "Buy put (IV cheap)" |
| neutral | > 50 | "Sell iron condor (IV rich)" |
| neutral | ≤ 50 | "Wait for clearer signal" |

**Note:** This is a simplified legacy function not connected to the main strategy pipeline in `options_data.py`, which does full contract selection with liquidity scoring.

---

## options_screener.py

### Purpose
Standalone per-ticker analysis combining technical momentum with options metrics. Used independently of the main dashboard pipeline.

### Functions

**`analyze_ticker(ticker)` → dict**
- Downloads 30-day price history
- Calls `calculate_historical_volatility()`
- Checks volume spike: today's volume vs 20-day average
- Computes 5-day price momentum
- Returns: ticker, current_price, hv_20d, volume_spike (bool), momentum_pct, direction

**`run_screener(tickers)` → DataFrame**
Applies `analyze_ticker` to a list, returns sorted DataFrame.

---

## rank_trades.py

### Purpose
Scores trade candidates with a simple "AI score" from 0–10.

### Function: `rank_trade(momentum, hv, volume_spike)` → float

```python
score = 0
if abs(momentum) > 2:  score += 3
if hv > 20:            score += 2
if volume_spike:       score += 3
if abs(momentum) > 5:  score += 2
return min(score, 10)
```

Legacy scoring function, not connected to the main strategy pipeline.

---

## predict_next_day.py

### Purpose
Standalone script that loads a trained model and the latest features CSV to produce a next-day range prediction for a single ticker.

### Function: `predict_for_ticker(ticker)` → dict

1. `joblib.load(f"models/{ticker}.pkl")`
2. `pd.read_csv(f"data/features/{ticker}.csv")`
3. Select last row, align to `feature_columns_`
4. Predict or use baseline
5. Clamp to floor/ceiling
6. Return prediction dict

Used for quick one-off checks without running the full API server.

---

## download_intraday.py

### Purpose
Downloads 2 years of daily OHLCV for all tickers in `TICKERS` and saves to CSVs.

### Function: `download_all(tickers, output_dir)` → None

Calls `yf.download(ticker, period="2y", interval="1d")` for each ticker and saves to `data/raw/{ticker}.csv`. This is the first step in the offline training pipeline before `build_features.py` and `train_models.py`.

---

## backtest.py

### Purpose
Quick backtesting utility using LightGBM to validate feature predictiveness.

### Libraries: `lightgbm`, `scikit-learn`, `pandas`

Loads a feature CSV, sets up `TimeSeriesSplit`, trains LightGBM with early stopping, reports MAE per fold. Used during feature development to quickly test whether a new feature improves predictions before adding it to the main training pipeline.

---

## Legacy / Experimental Files (not in main flow)

### calculate_gex.py
Duplicate of `gex_engine.py`. Safe to delete.

### detect_unusual_flow.py
Duplicate of `options_flow.py`. Safe to delete.

### ai_model.py
Placeholder model wrapper. Not used anywhere. Safe to delete.

### mega_double_random.py
Generates lottery number combinations using a weighted random selection with historical frequency analysis on Mega Millions draw data. Not connected to the options platform.

### mega_lstm_only.py
Uses a Keras LSTM network to predict lottery number patterns. Requires TensorFlow. Not connected to the options platform.

### options_dropdown_screener.py
A simple Streamlit UI wrapper around `options_screener.py`. Can be run standalone as `streamlit run options_dropdown_screener.py` for a simpler one-ticker analysis without the full dashboard.
