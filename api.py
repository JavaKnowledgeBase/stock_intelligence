from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd
from config import TICKERS
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from mega_double_random import generate_mega_numbers
from mega_lstm_only import generate_mega_numbers_deep_ai



app = FastAPI(title="Stock Range Prediction API")

templates = Jinja2Templates(directory="templates")

MODEL_DIR = "models"


@app.get("/")
def home():
    return {"message": "Stock Range Prediction API is running"}


@app.get("/dashboard/{ticker}", response_class=HTMLResponse)
def dashboard(request: Request, ticker: str):

    response = predict(ticker)

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "data": response}
    )


@app.get("/predict/{ticker}")
def predict(ticker: str):

    ticker = ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not trained yet")

    model = joblib.load(model_path)

    df = pd.read_csv(f"data/features/{ticker}.csv")
    latest = df.iloc[-1:]

    # Align features exactly as training
    X = latest.drop(columns=["high_t1", "low_t1"], errors="ignore")
    X = X.reindex(columns=model.feature_columns_, fill_value=0)

    prediction_pct = model.predict(X)[0]

    previous_close = float(latest["Close"].values[0])

    # Predicted prices
    predicted_range_dollars = previous_close * (prediction_pct / 100)
    predicted_high = previous_close + predicted_range_dollars / 2
    predicted_low = previous_close - predicted_range_dollars / 2

    # Predicted close (midpoint assumption)
    predicted_close = (predicted_high + predicted_low) / 2

    # Expected volatility
    expected_volatility = prediction_pct

    # Direction
    direction = "Bullish" if predicted_close > previous_close else "Bearish"

    # Confidence score (simple MAE-based proxy)
    confidence_score = max(0, 100 - model.training_mae_)

    # Historical comparison
    historical_avg_range = model.historical_avg_range_
    atr_comparison = prediction_pct / model.atr_avg_
    
    mega_numbers = generate_mega_numbers()
    
    mega_numbers_deep = generate_mega_numbers_deep_ai()

    


    return {
        "ticker": ticker,
        "previous_close": round(previous_close, 2),

        "predicted_high": round(predicted_high, 2),
        "predicted_low": round(predicted_low, 2),
        "predicted_close": round(predicted_close, 2),

        "predicted_range_percent": round(prediction_pct, 2),
        "predicted_range_dollars": round(predicted_range_dollars, 2),

        "expected_volatility_percent": round(expected_volatility, 2),

        "model_mae": round(model.training_mae_, 4),
        "confidence_score": round(confidence_score, 2),

        "historical_average_range_percent": round(historical_avg_range, 2),
        "atr_comparison_ratio": round(atr_comparison, 2),

        "direction": direction,
        "mega_millions_double_random": mega_numbers,
        "mega_millions_deep_ai": mega_numbers_deep,


    }
