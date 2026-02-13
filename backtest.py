
"""
backtest.py
Performs walk-forward time-series validation.
"""

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import pandas as pd
import lightgbm as lgb
from config import *

def run():
    df = pd.read_csv(f"{DATA_DIR}/features/{TICKERS[0]}.csv")
    X = df.drop(columns=["high_t1", "low_t1"])
    y = df["high_t1"]

    for tr, te in TimeSeriesSplit(5).split(X):
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[te])
        print("MAE:", mean_absolute_error(y.iloc[te], preds))
