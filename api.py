from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import pandas as pd
from config import TICKERS
from r2_storage import ensure_assets_available

app = FastAPI(title="Stock Range Prediction API")

MODEL_DIR = "models"
templates = Jinja2Templates(directory="templates")


def _should_use_baseline(model) -> bool:
    explicit_flag = getattr(model, "use_baseline_", None)
    if explicit_flag is not None:
        return bool(explicit_flag)

    training_mae = getattr(model, "training_mae_", None)
    baseline_mae = getattr(model, "baseline_mae_", None)
    if training_mae is None or baseline_mae is None:
        return False

    return float(baseline_mae) <= float(training_mae)


@app.get("/")
def home():
    return {"message": "Stock Range Prediction API is running", "dashboard": "/predictions"}


@app.on_event("startup")
def load_assets_on_startup():
    ensure_assets_available()


def _get_prediction_payload(ticker: str) -> dict:
    ensure_assets_available()
    ticker = ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not trained yet")

    model = joblib.load(model_path)

    df = pd.read_csv(f"data/features/{ticker}.csv")
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()].copy()

    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise HTTPException(status_code=500, detail="No valid feature rows available")

    latest = df.iloc[-1:]

    X = latest.drop(columns=["date"], errors="ignore")
    X = X.reindex(columns=model.feature_columns_, fill_value=0)

    if _should_use_baseline(model):
        prediction_pct = float(latest["hl_pct"].values[0])
        prediction_source = "baseline_fallback"
    else:
        prediction_pct = float(model.predict(X)[0])
        prediction_source = "trained_model"

    prediction_floor = getattr(model, "prediction_floor_", 0.5)
    prediction_ceiling = getattr(model, "prediction_ceiling_", 5.0)
    prediction_pct = max(min(prediction_pct, prediction_ceiling), prediction_floor)

    previous_close = float(latest["Close"].values[0])
    predicted_range_dollars = previous_close * (prediction_pct / 100)
    predicted_high = previous_close + predicted_range_dollars / 2
    predicted_low = previous_close - predicted_range_dollars / 2

    day_move = float(latest["oc_pct"].values[0])
    direction = "Bullish" if day_move >= 0 else "Bearish"

    confidence_score = max(
        0.0,
        min(
            100.0,
            100 * (1 - (getattr(model, "training_mae_", prediction_pct) / max(prediction_pct, 0.01))),
        ),
    )

    return {
        "ticker": ticker,
        "previous_close": round(previous_close, 2),
        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2),
        "predicted_range_percent": round(prediction_pct, 2),
        "predicted_range_dollars": round(predicted_range_dollars, 2),
        "direction": direction,
        "model_name": getattr(model, "model_name_", "unknown"),
        "model_mae": round(float(getattr(model, "training_mae_", 0.0)), 4),
        "baseline_mae": round(float(getattr(model, "baseline_mae_", 0.0)), 4),
        "prediction_source": prediction_source,
        "confidence_score": round(confidence_score, 2),
    }


@app.get("/dashboard/{ticker}", response_class=HTMLResponse)
def dashboard(request: Request, ticker: str):
    data = _get_prediction_payload(ticker)
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "data": data},
    )


@app.get("/predictions")
def all_predictions():
    results = []
    errors = []

    for ticker in TICKERS:
        try:
            results.append(_get_prediction_payload(ticker))
        except HTTPException as exc:
            errors.append({"ticker": ticker, "detail": exc.detail})
        except Exception as exc:
            errors.append({"ticker": ticker, "detail": str(exc)})

    results.sort(
        key=lambda item: (
            0 if item["prediction_source"] == "trained_model" else 1,
            -item["predicted_range_percent"],
        )
    )

    return {
        "count": len(results),
        "errors": errors,
        "predictions": results,
    }


@app.get("/predictions/ui", response_class=HTMLResponse)
def all_predictions_ui(request: Request):
    payload = all_predictions()
    return templates.TemplateResponse(
        "predictions.html",
        {"request": request, "payload": payload},
    )


@app.get("/predict/{ticker}")
def predict(ticker: str):
    return _get_prediction_payload(ticker)
