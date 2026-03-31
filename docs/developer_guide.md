# Developer Guide — Institutional Options Intelligence Platform

## Project Summary

A Python-based options intelligence platform built on Streamlit, with a parallel FastAPI backend for predictions and a nightly monitoring system. The core pipeline fetches live options and price data from Yahoo Finance, applies multi-factor technical scoring, selects liquid options contracts, and surfaces strategy ideas with ATR-based trade rules.

---

## Repository Structure

```
stock_intelligence_main_worktree/
│
├── dashboard.py                   # Streamlit UI — main entry point
├── options_data.py                # Core business logic (options + scoring)
├── build_features.py              # Technical feature engineering
├── train_models.py                # ML model training pipeline
├── api.py                         # FastAPI REST endpoints
├── nightly_monitor.py             # Nightly health check + email report
├── r2_storage.py                  # Cloudflare R2 sync
├── gex_engine.py                  # Gamma Exposure calculation
├── options_flow.py                # Unusual flow detection
├── config.py                      # Ticker lists, constants
│
├── data/
│   ├── features/                  # Pre-built feature CSVs per ticker
│   └── options_market_snapshots/  # Daily volume CSVs (auto-generated)
│
├── models/                        # Trained sklearn models as .pkl files
├── reports/nightly/               # JSON + CSV nightly reports
├── templates/                     # Jinja2 HTML templates for FastAPI
│
├── requirements.txt
├── .env                           # Credentials (R2, SMTP) — DO NOT COMMIT
├── run_server.ps1                 # Launch Streamlit
└── run_nightly_monitor.ps1        # Trigger nightly monitor
```

---

## Environment Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Cloudflare R2 (optional — for cloud model/data sync)
R2_BUCKET_NAME=your-bucket-name
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-key-id
R2_SECRET_ACCESS_KEY=your-secret-key

# Nightly monitor email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your-app-password
MONITOR_EMAIL_RECIPIENT=recipient@example.com
```

> **Security note:** Never commit `.env`. It is in `.gitignore`. Use app passwords for Gmail (not your account password).

### Running the Dashboard

```bash
streamlit run dashboard.py
```

Or with the PowerShell script:

```powershell
.\run_server.ps1
```

### Running the API

```bash
uvicorn api:app --reload --port 8000
```

### Running the Nightly Monitor Manually

```bash
python nightly_monitor.py
```

---

## Core Module: options_data.py

This is the single most important file. Everything the dashboard displays comes through here.

### In-Memory Cache

```python
_CACHE = {}          # dict: key → (timestamp, value)
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = 900
```

All yfinance calls are wrapped in `_get_cached` / `_set_cached`. This avoids hitting Yahoo's rate limits on every user interaction.

**Adding a new cached call:** wrap with `_cache_key(prefix, *args)` and the standard get/set pattern:

```python
def my_new_fetch(ticker):
    key = _cache_key("my_data", ticker)
    cached = _get_cached(key)
    if cached is not None:
        return _clone_frame(cached)
    result = ... fetch from yfinance ...
    return _set_cached(key, result.copy())
```

### Thread Safety

- **Never** use `yf.download()` in a parallel context. It shares internal session state and mixes up responses between threads. Always use `yf.Ticker(ticker).history()`.
- `_get_ticker(ticker)` caches the Ticker object — use it everywhere instead of creating `yf.Ticker()` directly.
- The `_CACHE_LOCK` is acquired only for the dict access operations, not during the yfinance download, to avoid serializing network calls.

### Adding a New Ticker Universe

Edit `config.py`:

```python
TICKERS = [...]              # Core tickers for options screener
MARKET_SCAN_TICKERS = [...]  # Wider universe for strategy + movers
```

Both lists are imported wherever needed. No other changes required.

### Adding a New Scoring Factor to Mover Rows

All factor adjustments are in `_build_mover_row_inner()` around lines 494–534. Pattern:

```python
# Compute your factor from latest (a pandas Series of features)
my_value = _safe_float(latest.get("my_feature", default_val))

# Compute upside/downside adjustment
my_adj_up = 1.5 if my_value > threshold else (-1.5 if my_value < low_threshold else 0.0)
my_adj_dn = -my_adj_up

# Add to score expressions (one_week_upside, one_week_downside, etc.)
one_week_upside = (... existing terms ... + my_adj_up)
one_week_downside = (... existing terms ... + my_adj_dn)
```

### Adding a New Column to the Strategy Table

1. Add the field to the return dict in `_build_strategy_recommendation()`
2. Add it to `base_cols` in `format_strategy_table()` in `dashboard.py`
3. Add the rename mapping in the same function

---

## Core Module: build_features.py

`build_features(df)` takes a raw OHLCV DataFrame (from yfinance) and returns a DataFrame with 60+ technical indicators. All indicators are computed with pandas rolling operations — no external TA libraries.

### Key Design Rules

- Returns an **empty DataFrame** (not None, not an exception) if input has fewer than 40 rows
- Drops rows where core columns are NaN: `Close`, `hl_pct`, `close_ret_1d`, `volatility_5d`
- Fills auxiliary columns with neutral values rather than dropping — prevents score leakage
- Does **not** modify the input DataFrame; works on a copy

### Adding a New Feature

Add it to the `daily` DataFrame inside `build_features()`:

```python
daily["my_feature"] = daily["Close"].rolling(10).mean() / daily["Close"] - 1
```

Then add it to the neutral-fill section near the bottom:

```python
neutral_fills = {
    ...,
    "my_feature": 0.0,  # neutral value if NaN
}
```

If the feature is required (not optional), add it to `core_cols` in the dropna section.

---

## Core Module: train_models.py

Trains one sklearn model per ticker. Called offline (not at runtime).

### Workflow

```bash
# 1. Download fresh OHLCV
python download_intraday.py

# 2. Build features CSVs
# (build_features is called internally by the train script)

# 3. Train models
python train_models.py
```

### Model Selection

Three candidates compete via TimeSeriesSplit cross-validation (3–5 folds). Lowest MAE wins. If the winning model's test MAE is worse than the naive baseline (predict yesterday's hl_pct), the model is flagged `use_baseline_=True` and the API will use the historical average instead.

### Model Metadata

Each `.pkl` file is a dict:

```python
{
    "model": sklearn_estimator,
    "model_name_": "RandomForestRegressor",
    "feature_columns_": [...],
    "training_mae_": float,
    "cross_val_mae_": float,
    "baseline_mae_": float,
    "use_baseline_": bool,
    "historical_avg_range_": float,
    "atr_avg_": float,
    "prediction_floor_": float,   # 10th percentile of training targets
    "prediction_ceiling_": float, # 90th percentile of training targets
    "feature_importance_top_": [...],  # top 10 features if applicable
}
```

### Adding a New Model Type

Add to the `candidates` list in `train_models.py`:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
candidates.append(("HGBR", HistGradientBoostingRegressor(...)))
```

---

## FastAPI Module: api.py

Provides REST access to the trained models. Runs separately from Streamlit on port 8000.

### Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | / | Health check JSON |
| GET | /predictions | All ticker predictions (JSON array) |
| GET | /predictions/ui | HTML table view |
| GET | /predict/{ticker} | Single ticker prediction JSON |
| GET | /dashboard/{ticker} | Jinja2 HTML page |

### Prediction Payload

```json
{
  "ticker": "AAPL",
  "previous_close": 213.49,
  "predicted_high": 216.85,
  "predicted_low": 210.13,
  "predicted_range_pct": 3.14,
  "direction": "up",
  "model_name": "RandomForestRegressor",
  "mae": 0.82,
  "confidence_score": 0.76
}
```

---

## R2 Storage: r2_storage.py

On startup, `ensure_assets_available()` is called in `dashboard.py` before anything else. It:

1. Checks if R2 credentials are present in `.env`
2. If yes: syncs `models/` and `data/` from R2 bucket
3. If no or R2 fails: uses whatever is already local
4. If R2 fails AND local is missing: raises RuntimeError

This means the dashboard can run on a fresh deployment with no local models as long as R2 is configured.

### Uploading to R2 (after retraining)

There is no auto-upload in the current code. After retraining:

```python
import boto3
s3 = boto3.client("s3",
    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)
s3.upload_file("models/AAPL.pkl", os.getenv("R2_BUCKET_NAME"), "models/AAPL.pkl")
```

---

## Nightly Monitor: nightly_monitor.py

Designed to run via cron or Windows Task Scheduler after market close (8 PM ET).

### Scheduling (Windows Task Scheduler / PowerShell)

```powershell
# run_nightly_monitor.ps1
python C:\path\to\nightly_monitor.py
```

### What It Checks

- Forecast row count >= 5 (MIN_FORECAST_ROWS)
- Strategy row count >= 5 (MIN_STRATEGY_ROWS)
- Median confidence >= 70% (RECOMMENDED_CONFIDENCE)
- Coverage dropped > 30% vs previous run

Recommendations are generated as plain-text strings and included in the email.

### Output Files

```
reports/nightly/20260331_200000.json             # full summary
reports/nightly/20260331_200000_forecast.csv     # gainers + losers
reports/nightly/20260331_200000_strategy.csv     # strategy ideas
```

---

## GEX Engine: gex_engine.py

```python
def calculate_gex(chain_df):
    gex = chain_df["gamma"] * chain_df["openInterest"] * 100 * chain_df["strike"]
    return gex.groupby(chain_df["strike"]).sum()
```

Returns a Series indexed by strike. Plotted as a bar chart in the Options Chain Explorer tab.

---

## Unusual Flow Detection: options_flow.py

```python
def detect_unusual_flow(chain_df):
    return chain_df[
        (chain_df["volume"] > 500) &
        (chain_df["volume"] / chain_df["openInterest"].replace(0, np.nan) > 2)
    ].sort_values("volume", ascending=False)
```

Contracts where today's volume is more than twice the open interest — potential institutional "prints".

---

## Common Development Tasks

### Running a Quick Test of the Scoring Pipeline

```python
from options_data import build_market_movers_table
from config import MARKET_SCAN_TICKERS

df = build_market_movers_table(MARKET_SCAN_TICKERS[:5])  # just 5 tickers
print(df[["ticker", "close", "one_week_view", "one_week_score"]])
```

### Testing Strategy Contract Selection

```python
from options_data import _pick_strategy_contract

result = _pick_strategy_contract("AAPL", "call", close_price=213.0)
print(result)
```

### Inspecting a Trained Model

```python
import joblib
model_data = joblib.load("models/AAPL.pkl")
print(model_data["model_name_"])
print(model_data["feature_importance_top_"])
print(f"Test MAE: {model_data['training_mae_']:.3f}")
print(f"Uses baseline: {model_data['use_baseline_']}")
```

### Clearing the In-Memory Cache

```python
from options_data import _CACHE, _CACHE_LOCK
with _CACHE_LOCK:
    _CACHE.clear()
```

---

## Code Conventions

- **No ORM, no DB** — all persistent state is flat files (CSV, pkl, JSON)
- **No global mutable state** except `_CACHE` (protected by `_CACHE_LOCK`)
- All public functions return a DataFrame or a (DataFrame, dict) tuple — never None without documentation
- `_safe_float(value, default, minimum, maximum)` — always use for any yfinance-sourced numeric
- Prefer `_clean_numeric_columns(df, defaults_dict)` for batch coercion over inline `.fillna()`
- Never call `yf.download()` — only `_get_ticker(ticker).history()`
- Thread-submitted functions (`_build_mover_row`, `_build_strategy_recommendation`) catch all exceptions internally and return None — the caller skips None results

---

## Known Limitations & Tech Debt

| Issue | Location | Notes |
|---|---|---|
| No retry/backoff on yfinance 429 | options_data.py | Rate-limited calls fail immediately; rate_limited flag is set in diagnostics |
| Double-download race condition | options_data.py:get_price_history | Two threads with same key both download before caching; harmless but wasteful |
| Legacy duplicate files | calculate_gex.py, detect_unusual_flow.py | Same logic as gex_engine.py / options_flow.py; safe to delete |
| Experimental code in repo | mega_double_random.py, mega_lstm_only.py | Lottery number generators; not connected to anything |
| No upload path for R2 | r2_storage.py | Manual boto3 call required after retraining models |
| Credentials in .env tracked by git history | .env | Rotate keys if repo was ever public |
