# File Reference: api.py

## Purpose

A FastAPI REST service that exposes next-day range predictions from trained ML models. It runs separately from the Streamlit dashboard on port 8000. Intended for programmatic access — other applications can query predictions without running the full Streamlit UI.

---

## Libraries Used

| Library | Why |
|---|---|
| `fastapi` | REST framework: routing, request/response, automatic OpenAPI docs |
| `uvicorn` | ASGI server to run FastAPI |
| `jinja2` (via fastapi.templating) | HTML template rendering for browser-friendly views |
| `fastapi.staticfiles` | Serve static assets if any |
| `joblib` | Load trained `.pkl` model files |
| `pandas` | Read feature CSVs, manipulate prediction data |
| `numpy` | Numeric operations for prediction clamping |
| `pathlib` | Locate model and feature files |
| `r2_storage` | `ensure_assets_available()` on startup |
| `config` | `TICKERS`, `DATA_DIR` |

---

## Startup

```python
@app.on_event("startup")
async def startup_event():
    ensure_assets_available()
```

Downloads models and data from R2 before the first request is served, ensuring a fresh deployment has everything it needs.

---

## Endpoints

### `GET /`
**Health check.**
Returns: `{"status": "ok", "tickers": [...], "model_count": N}`

---

### `GET /predictions`
**All ticker predictions as JSON.**
Iterates `TICKERS`, calls `_get_prediction_payload(ticker)` for each, collects results.

Returns:
```json
[
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
  },
  ...
]
```

---

### `GET /predictions/ui`
**HTML table view of all predictions.**
Rendered with Jinja2 template `templates/predictions.html`.

---

### `GET /predict/{ticker}`
**Single ticker prediction.**
Returns the same payload structure as one element from `/predictions`.
Returns 404 if ticker not in TICKERS or model file not found.

---

### `GET /dashboard/{ticker}`
**Single ticker HTML dashboard.**
Rendered with Jinja2 template `templates/dashboard.html`.
Passes the full prediction payload plus additional metadata (feature importance, model metadata) to the template.

---

## Core Function: `_get_prediction_payload(ticker)` → dict

**Steps:**
1. Load model: `joblib.load(f"models/{ticker}.pkl")`
2. Load latest features: `pd.read_csv(f"data/features/{ticker}.csv")`
3. Select the last row of features matching `model_data["feature_columns_"]`
4. If `use_baseline_=True`: use `historical_avg_range_` directly
5. Else: `predicted_range = model.predict(X_latest)[0]`
6. Clamp: `predicted_range = clip(predicted_range, prediction_floor_, prediction_ceiling_)`
7. Compute `predicted_high = previous_close + (range/2)`, `predicted_low = previous_close - (range/2)`
8. Compute `direction`: compare predicted Close direction (heuristic from RSI or MACD features)
9. Compute `confidence_score` from model metadata (inverse of normalised MAE)

**Returns:** Full prediction dict including previous_close, predicted_high, predicted_low, predicted_range_pct, direction, model_name, mae, confidence_score.

---

## Model File Contract

Every `.pkl` file must contain these keys (set by `train_models.py`):

```python
{
    "model":                 fitted sklearn estimator (or None if use_baseline_),
    "model_name_":           str,
    "feature_columns_":      list[str],
    "use_baseline_":         bool,
    "historical_avg_range_": float,
    "prediction_floor_":     float,
    "prediction_ceiling_":   float,
    "training_mae_":         float,
}
```

---

## Running the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## Relationship to Dashboard

The API and dashboard are independent. The dashboard uses `options_data.py` directly (live yfinance data). The API uses pre-trained models and saved feature CSVs. They share: `config.py`, `r2_storage.py`, and `build_features.py` (indirectly via saved CSVs).
