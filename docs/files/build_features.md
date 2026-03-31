# File Reference: build_features.py

## Purpose

Transforms raw OHLCV (Open, High, Low, Close, Volume) price history into a rich set of 60+ technical features used for market mover scoring and ML model training. This is a pure feature engineering module — no external data fetched, no side effects, no persistence.

---

## Libraries Used

| Library | Why |
|---|---|
| `pandas` | All rolling calculations, index operations, DataFrame construction |
| `numpy` | `np.nan`, `np.log`, `np.sqrt`, `np.where` for vectorised operations |

No external TA library (e.g., ta-lib) is used — all indicators are hand-implemented with pandas rolling operations.

---

## Entry Point

### `build_features(df)` → DataFrame

**Input:** Raw OHLCV DataFrame from yfinance (single-ticker).

**Output:** DataFrame with all original OHLCV columns plus 60+ derived features.

**Returns empty DataFrame if:**
- Input is None or empty
- Input has fewer than 40 rows (insufficient for rolling windows)

**Key pre-processing:**
- `_flatten_columns(df)` — handles yfinance MultiIndex columns
- `df.loc[:, ~df.columns.duplicated()]` — drops duplicate column names
- `_coerce_numeric_columns(df)` — forces OHLCV to float
- DatetimeIndex coercion + sort
- `dropna(subset=["Open","High","Low","Close","Volume"])` — removes invalid rows

---

## Helper Functions

### `_flatten_columns(df)`
Identical logic to `_flatten_df_columns` in options_data.py. Handles MultiIndex by finding the level containing price column names. Fallback: take `col[0]` from each tuple.

### `_coerce_numeric_columns(df)`
Iterates over OHLCV columns, coerces each to numeric with `errors="coerce"`. Handles the case where yfinance returns a column as a DataFrame (takes `.iloc[:,0]` first).

---

## Feature Groups

### Date Features
```
date             = index.strftime("%Y-%m-%d")
day_of_week      = index.dayofweek  (0=Mon, 4=Fri)
day              = index.day
month            = index.month
quarter          = index.quarter
is_month_start   = bool
is_month_end     = bool
```

### Price Structure
```
hl_pct           = (High − Low) / Close × 100       ← primary target for ML model
oc_pct           = (Close − Open) / Open × 100
gap_pct          = (Open − PrevClose) / PrevClose × 100
intraday_position= (Close − Low) / (High − Low)     ← 0=closed at low, 1=closed at high
upper_shadow_pct = (High − max(Open,Close)) / Close × 100
lower_shadow_pct = (min(Open,Close) − Low) / Close × 100
```

### Returns
```
close_ret_1d     = (Close / Close.shift(1) − 1) × 100
close_ret_3d     = (Close / Close.shift(3) − 1) × 100
close_ret_5d     = (Close / Close.shift(5) − 1) × 100
volume_ret_1d    = (Volume / Volume.shift(1) − 1) × 100
```

### Volatility
```
volatility_5d    = hl_pct.rolling(5).mean()
volatility_10d   = hl_pct.rolling(10).mean()
volatility_std_5d= hl_pct.rolling(5).std()
volatility_std_10d= hl_pct.rolling(10).std()
atr_14           = EWM(span=14).mean() of True Range
                   True Range = max(H−L, |H−PrevC|, |L−PrevC|)
```

### Moving Averages
```
ma_5, ma_10, ma_20   = Close.rolling(n).mean()
ema_5, ema_10, ema_20 = Close.ewm(span=n, adjust=False).mean()

dist_ma_5_pct   = (Close − ma_5)  / ma_5  × 100
dist_ma_10_pct  = (Close − ma_10) / ma_10 × 100
dist_ma_20_pct  = (Close − ma_20) / ma_20 × 100
dist_ema_10_pct = (Close − ema_10)/ ema_10 × 100
dist_ema_20_pct = (Close − ema_20)/ ema_20 × 100
```

### Bollinger Bands
```
bb_upper = ma_20 + 2 × rolling_std_20
bb_lower = ma_20 − 2 × rolling_std_20
bb_pct   = (Close − bb_lower) / (bb_upper − bb_lower)   ← 0=at lower, 1=at upper
```

### RSI (14-period, Wilder's smoothing)
```
delta = Close.diff()
gain  = delta.clip(lower=0)
loss  = (−delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
rs    = avg_gain / avg_loss.replace(0, np.nan)
rsi_14= (100 − 100/(1+rs)).clip(0, 100)
```

### MACD (12/26/9 EMA)
```
ema_12        = Close.ewm(span=12, adjust=False).mean()
ema_26        = Close.ewm(span=26, adjust=False).mean()
macd          = ema_12 − ema_26
macd_signal   = macd.ewm(span=9, adjust=False).mean()
macd_hist     = macd − macd_signal
```

### ADX (14-period Average Directional Index)
```
plus_dm  = max(High − PrevHigh, 0) where High − PrevHigh > PrevLow − Low
minus_dm = max(PrevLow − Low, 0)  where PrevLow − Low > High − PrevHigh
(zero if tie or same direction as opposite)

Smoothed with EWM span=14:
plus_di_14  = 100 × EWM(plus_dm) / atr_14
minus_di_14 = 100 × EWM(minus_dm) / atr_14
dx          = |plus_di − minus_di| / (plus_di + minus_di) × 100
adx_14      = dx.ewm(span=14, adjust=False).mean()
```

### Volume Analysis
```
volume_ratio_5  = Volume / Volume.rolling(5).mean()
volume_ratio_20 = Volume / Volume.rolling(20).mean()

vol_direction_ratio: up-day volume vs down-day volume over 10 days
  up_vol_10d   = sum of Volume where Close > PrevClose (rolling 10)
  dn_vol_10d   = sum of Volume where Close ≤ PrevClose (rolling 10)
  vol_direction_ratio = up_vol_10d / dn_vol_10d (NaN if dn_vol = 0)

trend_consistency_10d = fraction of last 10 days with positive Close return
                      = rolling 10-day mean of (close_ret_1d > 0).astype(int)
```

### OBV (On-Balance Volume)
```
OBV: cumulative sum of Volume × sign(Close − PrevClose)
  +Volume if Close > PrevClose
  −Volume if Close < PrevClose
  0 if unchanged

obv_slope_10d = (OBV / OBV.shift(10) − 1) × 100  ← 10-day % change
```

### Relative Position
```
high_52w         = Close.rolling(252).max()
low_52w          = Close.rolling(252).min()
pct_from_52w_high= (Close − high_52w) / high_52w × 100   ← negative value
pct_from_52w_low = (Close − low_52w) / low_52w  × 100    ← positive value
```

### MA Alignment
```
ma_alignment = +1 if Close > ma_10 > ma_20 (bullish stack)
              = −1 if Close < ma_10 < ma_20 (bearish stack)
              =  0 otherwise (mixed)
```

### Lagged Features (for ML)
For each of: hl_pct, oc_pct, gap_pct, close_ret, volume_ratio_5 — lags 1, 2, 3, 5:
```
hl_pct_lag_1 = hl_pct.shift(1)
hl_pct_lag_2 = hl_pct.shift(2)
...
```

---

## NaN Handling Strategy

**Core columns** (drop row if NaN):
```python
core_cols = ["Close", "hl_pct", "close_ret_1d", "volatility_5d"]
daily = daily.dropna(subset=core_cols)
```

**Auxiliary columns** (fill with neutral value):
```python
neutral_fills = {
    "close_ret_3d": 0.0, "close_ret_5d": 0.0,
    "gap_pct": 0.0, "intraday_position": 0.5,
    "volatility_10d": "ffill",
    "rsi_14": 50.0, "macd": 0.0, "macd_hist": 0.0,
    "adx_14": 20.0, "ma_alignment": 0,
    "volume_ratio_5": 1.0, "volume_ratio_20": 1.0,
    "obv_slope_10d": 0.0, "bb_pct": 0.5,
    "pct_from_52w_high": -50.0, "pct_from_52w_low": 50.0,
    "trend_consistency_10d": 0.5,
    "vol_direction_ratio": 1.0,
    # ... all lag features filled with 0.0
}
```

The fill-with-neutral approach ensures that a ticker with less than 252 days of history (needed for 52-week stats) still produces valid rows — the 52W feature just uses a neutral default. This prevents premature exclusion from scoring.

---

## Usage Context

1. **In options_data.py / _build_mover_row_inner():**
   Called on 1-year history. `latest = features.iloc[-1]` extracts today's features for scoring.

2. **In train_models.py:**
   Called on 2-year history. Target is `hl_pct.shift(-1)` (next day's range). All feature columns except date become model inputs.

3. **In api.py (via saved features):**
   Features are pre-computed and saved to `data/features/{ticker}.csv`. The API loads the latest row and passes to the trained model.
