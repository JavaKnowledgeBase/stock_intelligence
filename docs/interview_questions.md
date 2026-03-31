# Interview Questions Reference — Institutional Options Intelligence Platform

This document covers likely interview questions based on the technologies, patterns, and algorithms used in this project. Answers are tailored to what is actually implemented here.

---

## Python & General Programming

**Q: Explain the threading model used in this project and why it was chosen.**

A: The project uses `concurrent.futures.ThreadPoolExecutor` with up to 8–10 workers. This is appropriate because the bottleneck is I/O (network calls to Yahoo Finance), not CPU. Threads release the GIL during I/O, so multiple yfinance requests run concurrently without GIL contention. Each worker calls `_build_mover_row()` or `_build_strategy_recommendation()` for a different ticker — fully independent work units with no shared mutable state except the cache dict, which is protected by a `threading.Lock`.

---

**Q: What is the GIL and why doesn't it block parallelism here?**

A: The Global Interpreter Lock prevents multiple Python threads from executing Python bytecode simultaneously. However, during I/O operations (network requests, file reads), the GIL is released, allowing other threads to run. Since this project's main workload is fetching data from Yahoo Finance over HTTP, threads are mostly waiting on I/O, not executing Python — so the GIL is not a bottleneck. For CPU-bound work (like training sklearn models), `ProcessPoolExecutor` or multiprocessing would be more appropriate.

---

**Q: Describe the caching architecture and its trade-offs.**

A: A thread-safe in-memory dict (`_CACHE`) stores (timestamp, value) pairs keyed by a tuple of (prefix, ticker, period, interval). Cache misses trigger a yfinance download; hits return a `.copy()` to prevent callers from mutating cached data. TTL is 900 seconds. Trade-offs: fast (no serialisation overhead), simple, but lost on process restart. A Redis or disk cache would persist across restarts. The current design has a classic TOCTOU race — two threads can both find a cache miss for the same key and both download, but since results are the same data, it's harmless.

---

**Q: Why was `yf.download()` replaced with `yf.Ticker().history()`?**

A: `yf.download()` has a known thread-safety bug when called from multiple threads simultaneously — it uses shared internal session state and can return one ticker's HTTP response attributed to a different ticker. This manifested as multiple tickers showing identical prices. `yf.Ticker(ticker).history()` uses per-ticker download context, making it safe for concurrent use. The `yf.Ticker` objects are also cached to avoid re-creation overhead.

---

**Q: What is a race condition and where could one occur in this code?**

A: A race condition is when the correctness of a program depends on the relative timing of thread execution. In `get_price_history()`, between the `_get_cached()` call returning None and `_set_cached()` storing the result, another thread with the same key can also call `_get_cached()` (also getting None) and start a duplicate download. Both threads will eventually store valid data under the same key, so the result is correct — just wasteful. A double-checked locking pattern (re-check inside the lock after acquiring it) would eliminate the duplicate download.

---

**Q: Explain the use of `_safe_float()` — why not just cast directly?**

A: yfinance data frequently contains NaN, None, infinity, or strings where numbers are expected. Direct `float(value)` would raise exceptions or silently produce `inf`. `_safe_float` uses `pd.to_numeric(value, errors="coerce")` to handle all these cases, checks `np.isfinite()`, falls back to a configurable `default`, and optionally enforces `minimum`/`maximum` bounds. This defensive pattern prevents a single bad data point from crashing a scan of 70 tickers.

---

**Q: What is a context manager and where is it used here?**

A: A context manager implements `__enter__` and `__exit__` protocols, typically used with `with` statements. Here: `with _CACHE_LOCK:` acquires a `threading.Lock` on entry and releases it on exit, even if an exception occurs. `with ThreadPoolExecutor(...) as pool:` ensures threads are properly joined on exit. Both guarantee cleanup regardless of exceptions.

---

## pandas & Data Manipulation

**Q: What is a MultiIndex DataFrame and how is it handled here?**

A: A MultiIndex DataFrame has a hierarchical column structure. Newer yfinance versions return price data with columns like `('Close', 'AAPL'), ('Open', 'AAPL'), ...` instead of just `'Close', 'Open', ...`. `_flatten_df_columns()` detects MultiIndex, iterates through levels to find whichever level contains recognised price column names (`{'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}`), and uses that level as the flat column index. Fallback: take `col[0]` from each tuple. This handles both old and new yfinance API formats.

---

**Q: Explain `.shift()` and when you'd use it in a time-series context.**

A: `series.shift(n)` moves values forward by n rows (positive n) or backward (negative n). Used here to compute: `close_ret_1d = (Close - Close.shift(1)) / Close.shift(1)` (compare today to yesterday), `prev_close = Close.shift(1)` (align yesterday's close with today's row), and in `train_models.py`: `target = hl_pct.shift(-1)` (predict the NEXT day's range by shifting the target one row back relative to features). Critical pattern in time-series ML — shifting forward by 1 to avoid data leakage.

---

**Q: What is the difference between `fillna()` and `dropna()` and when do you use each?**

A: `dropna()` removes rows with missing values — used here for core OHLCV columns where absence of data means the row is unusable. `fillna()` replaces NaN with a specified value — used for auxiliary/derived features like RSI, ADX, OBV slope where a neutral default (50 for RSI, 0 for returns) is meaningful and preserves the row for scoring. The distinction is: does missing data mean "this row is corrupt" (drop) or "I don't have context for this indicator yet" (fill with neutral)?

---

**Q: Explain `groupby()` + `pivot_table()` as used in the volume trend calculation.**

A: In `build_market_volume_table()`, after loading 7 days of daily snapshot CSVs, a `pivot_table()` is computed: `index=ticker, columns=date, values=one_day_volume, aggfunc="sum"`. This produces a matrix where each row is a ticker and each column is a date, filled with daily volume. The trend is then calculated by comparing the mean of the first half of dates to the mean of the second half — an increasing trend if the recent average exceeds the earlier average by >5%.

---

**Q: What is `.clip()` and where is it important here?**

A: `clip(lower, upper)` constrains values to a range. Used extensively: `np.clip(adx_14 / 35.0, 0.65, 1.15)` ensures the ADX confidence multiplier stays between 0.65 and 1.15 regardless of extreme ADX values. `(18 - spread_pct).clip(lower=0)` ensures the spread score never goes negative. Without clipping, extreme inputs (ADX of 100, spread of 200%) would produce unbounded scores that would overwhelm the other terms.

---

## Options & Finance Concepts

**Q: What is open interest vs volume?**

A: Volume is the number of contracts traded today — resets to zero each day. Open interest is the total number of outstanding contracts — cumulative and changes when positions are opened or closed. High OI with low volume means many existing positions but little activity today. High volume with low OI means lots of new trading relative to existing positions — a potential signal of new institutional intent. The unusual flow detector looks for `volume / OI > 2`.

---

**Q: What is implied volatility (IV) and what does IV/HV ratio indicate?**

A: Implied volatility is the market's forward-looking estimate of volatility, extracted from option prices using an options pricing model. Historical volatility (HV) is the realised standard deviation of returns over a lookback period. The IV/HV ratio compares expected vs realised volatility. IV/HV > 1.5 means options are expensive relative to recent realised moves — options sellers are favoured. IV/HV < 0.8 means options are cheap — options buyers are favoured.

---

**Q: Explain the concept of gamma exposure (GEX) and its significance.**

A: Gamma is the rate of change of delta with respect to the underlying price. Dealers (market makers) who sell options must hedge by dynamically adjusting their stock position (delta hedging). GEX = Gamma × Open Interest × 100 × Strike represents the aggregate hedging pressure at each strike. Positive GEX: dealers are long gamma and act as a stabilising force (buying dips, selling rips). Negative GEX: dealers are short gamma and amplify moves. The strike with largest positive GEX often acts as a magnet for price during the day.

---

**Q: What is ATR and how is it used here for position sizing?**

A: Average True Range measures volatility as the average of the greatest of: (High−Low), |High−PrevClose|, |Low−PrevClose|. Here it's computed as an EMA over 14 days. In strategy recommendations, ATR is used for position sizing: stop loss = entry − 1.5×ATR (15–30% of option price), first target = entry + 2×ATR (20–35%), second target = entry + 3×ATR (30–50%). This adapts the risk parameters to each stock's actual volatility rather than using fixed percentages.

---

**Q: What is the significance of RSI thresholds in the scoring model?**

A: RSI (Relative Strength Index) measures momentum on a 0–100 scale. The model applies an additive bonus of +2.0 for calls when RSI is in the 50–70 range (trending up but not overextended) and a penalty of −3.0 when RSI > 75 (overbought — mean reversion risk). For puts, the ideal zone is 30–50. The asymmetric penalty (−3 for extremes, +2 for ideal) reflects that overly extended moves carry higher reversal risk than the directional reward justifies.

---

**Q: Explain the market regime filter and why it suppresses signals.**

A: The regime is determined by SPY's 5-day return and 20-day MA position: bullish if ret_5d > 1% and price > MA20, bearish otherwise. In a bullish regime, put signals are only kept if their strategy score is in the top 40% of all signals — the idea is that low-conviction counter-trend bets are noise in a trending market. High-conviction counter-trend signals (top 40%) can still survive as hedges or genuine reversals. This prevents the system from flooding the strategy table with low-quality contrarian bets when the market has clear directional momentum.

---

## Machine Learning

**Q: What is TimeSeriesSplit and why must it be used instead of regular cross-validation for financial data?**

A: TimeSeriesSplit splits data sequentially — training on the past, validating on a future window — preserving temporal order. Regular k-fold cross-validation randomly assigns rows to folds, which would allow the model to "see the future" during training (data leakage). In financial time series, this produces optimistically biased validation scores that don't reflect real out-of-sample performance. TimeSeriesSplit ensures the validation set always comes after the training set chronologically.

---

**Q: What is data leakage and how is it prevented in build_features.py?**

A: Data leakage occurs when information from the test/future period influences training. The main risk here is the target: `hl_pct.shift(-1)` (next day's range). Features must use only past/present data. Using `.rolling(n).mean()` with default `min_periods=1` computes from past n days only — no lookahead. The `shift(-1)` target is only used in `train_models.py` after the features are fully computed, and the last row (which would have NaN as the target) is dropped before training.

---

**Q: Why are three model types compared (RandomForest, ExtraTree, GradientBoosting)?**

A: Each has different inductive biases: RandomForest aggregates many decorrelated trees by bootstrap sampling and feature subsampling — good general baseline. ExtraTrees (ExtraTreesRegressor) adds random feature thresholds, further reducing variance at the cost of slight bias increase. GradientBoosting (with absolute_error loss) is more sensitive to the training sequence, can capture complex patterns, but risks overfitting. The competition via TimeSeriesSplit MAE selects the model that generalises best to held-out future periods for each specific ticker's historical pattern.

---

**Q: What is MAE (Mean Absolute Error) and why is it preferred over MSE here?**

A: MAE is the average absolute difference between predicted and actual values — it treats all errors equally. MSE squares errors, making it more sensitive to outliers. In options/financial data, outlier days (earnings, macro events) are real events and their prediction errors should count proportionally, not be overweighted quadratically. The baseline comparison (predicted vs. naive "same as yesterday") uses MAE: if the trained model can't beat a naive predictor, the platform uses the historical average instead.

---

**Q: What does `use_baseline_` mean and how is it applied?**

A: If the trained model's test MAE exceeds the naive baseline MAE (predicting yesterday's hl_pct), it means the model found patterns in training that don't generalise. The `use_baseline_=True` flag is set, and the API uses `historical_avg_range_` (mean of all training targets) as the prediction instead. The prediction is still clamped between `prediction_floor_` (10th percentile of training data) and `prediction_ceiling_` (90th percentile) to avoid extreme outlier predictions.

---

## System Design & Architecture

**Q: How does the strategy pipeline scale with more tickers?**

A: The pipeline is embarrassingly parallel at the ticker level. Adding tickers to `MARKET_SCAN_TICKERS` increases the number of parallel workers needed. The current limit is `_MAX_WORKERS = 8`. Adding 50 more tickers would primarily increase the wall-clock time if workers = 8, or it can be increased up to the yfinance rate limit. The bottleneck is yfinance: more than ~20 simultaneous requests consistently triggers 429 rate limiting. The rate-limited error detection and diagnostics system surfaces this gracefully.

---

**Q: Why is R2 used instead of a traditional database?**

A: The data accessed here (trained models as .pkl files, feature CSVs) is read-heavy, append-only, and file-structured — not relational. R2 (Cloudflare's S3-compatible object storage) is cheap, globally accessible, and maps naturally to the filesystem layout used locally. A database would add complexity for no benefit: there are no joins, no transactions, and no concurrent writes to the same record. The R2 integration is purely for deployment: models trained locally can be uploaded to R2 and synced to any server running the dashboard.

---

**Q: How does the confidence score work and what are its inputs?**

A: `confidence = clip(ADX/35, 0.65, 1.15) × clip(vol_ratio/1.5, 0.75, 1.15) × clip(1 − |RSI−50|/100, 0.70, 1.00)`

- ADX/35 captures trend strength — a strong trend (ADX=35) produces a multiplier of 1.0; very weak trends (ADX=20) give 0.65
- vol_ratio/1.5 captures volume confirmation — above-average volume (ratio=1.5) gives 1.0; flat volume gives 0.75
- 1 − |RSI−50|/100 captures momentum alignment — RSI=50 (balanced) gives 1.0; extremes like RSI=80 give 0.70

All three are multiplicative — a strong trend with weak volume confirmation gets partially penalised. The minimum floor (0.60) rejects signals where all three indicators are weak.

---

**Q: Describe a scenario where this system would produce bad output and how to detect it.**

A: Three main failure scenarios:

1. **Yahoo rate limiting (429)**: Seen when running repeatedly or outside market hours. Detected via `_is_rate_limited_error()` message parsing. Diagnostics show `rate_limited=True` and the status message explains it.

2. **Stale/zero volume on weekends**: yfinance returns 0 volume for all contracts. Every contract fails the volume≥50 filter. Result: "no_contracts" status. Detectable from the nightly monitor's row count check.

3. **Deep ITM strike selection (now fixed)**: Before the strike range filter, contracts with years of accumulated OI could pass the liquidity filter at extreme strikes. Now prevented by the hard ±15% range constraint.
