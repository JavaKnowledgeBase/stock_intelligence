# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 07:09:49 2026

@author: rkafl
"""

# config.py  single source of truth

"""
Central configuration file.
Change values here — the rest of the project adapts automatically.
"""

# Stocks included in the pooled model
# TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

TICKERS = ["NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "AVGO", 
           "TSLA", "WMT", "MU", "BAC", "ORCL", 
           "NFLX", "AMD", "CSCO", "PLTR", "INTC", "VZ", "T", "UBER", 
           "PFE", "CMCSA", "HOOD", "WBD", "CCL", "HPE", "SMCI"]

# How many calendar days of intraday data to download
# (Yahoo limits intraday history per request)
INTRADAY_DAYS = 360

# Market hours (US equities)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# Base data directory
DATA_DIR = "C:\\Users\\rkafl\\OneDrive\\Desktop\\data\\intraday"


"""
build_features.py
Builds daily ML features from intraday data.
"""

import pandas as pd
import os

def build_features(symbol):
    df = pd.read_csv(f"{DATA_DIR}/intraday/{symbol}.csv", index_col=0, parse_dates=True)

    daily = df.groupby(df.index.date).agg({
        "Open": "first",
        "Close": "last",
        "High": "max",
        "Low": "min",
        "Volume": "sum"
    })

    daily["range"] = daily["High"] - daily["Low"]
    daily["std_5m"] = df.groupby(df.index.date)["Close"].std()
    daily["ret"] = daily["Close"].pct_change()
    daily["atr_5"] = daily["range"].rolling(5).mean()
    daily["atr_10"] = daily["range"].rolling(10).mean()
    daily["sma_20"] = daily["Close"].rolling(20).mean()
    daily["dist_sma20"] = (daily["Close"] - daily["sma_20"]) / daily["Close"]

    daily["high_t1"] = (daily["High"].shift(-1) - daily["Close"]) / daily["Close"]
    daily["low_t1"] = (daily["Low"].shift(-1) - daily["Close"]) / daily["Close"]
    daily["ticker"] = symbol

    os.makedirs(f"{DATA_DIR}/features", exist_ok=True)
    daily.dropna().to_csv(f"{DATA_DIR}/features/{symbol}.csv")

def run():
    for t in TICKERS:
        build_features(t)
        

"""
backtest.py
Performs walk-forward time-series validation.
"""

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import pandas as pd
import lightgbm as lgb

def run():
    df = pd.read_csv(f"{DATA_DIR}/features/{TICKERS[0]}.csv")
    X = df.drop(columns=["high_t1", "low_t1"])
    y = df["high_t1"]

    for tr, te in TimeSeriesSplit(5).split(X):
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        print("MAE:", mean_absolute_error(y.iloc[te], preds))
        
        

"""
predict_next_day.py
Generates next-day high/low prediction.
"""

import joblib

def run():
    models = joblib.load("models.pkl")
    df = pd.read_csv(f"{DATA_DIR}/features/{TICKERS[0]}.csv").iloc[-1:]
    close = df["Close"].values[0]

    X = pd.get_dummies(df.drop(columns=["high_t1", "low_t1"]))
    X = X.reindex(columns=models["high_q90"].feature_name_, fill_value=0)

    high = close * (1 + models["high_q90"].predict(X)[0])
    low = close * (1 + models["low_q10"].predict(X)[0])

    print(f"Predicted range: {low:.2f} – {high:.2f}")
