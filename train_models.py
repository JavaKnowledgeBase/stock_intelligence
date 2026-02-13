"""
train_models.py

Trains a machine learning model using engineered features.
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def train_model(ticker: str, df: pd.DataFrame):
    """
    Train a regression model to predict next-day price range.

    Parameters
    ----------
    ticker : str
        Stock symbol (used for logging only)
    df : pd.DataFrame
        Feature-engineered DataFrame

    Returns
    -------
    model : trained ML model
    """

    # -------------------------------
    # Target: next-day high-low range
    # -------------------------------
    y = df["hl_pct"].shift(-1).dropna()
    # X = df.iloc[:-1].drop(columns=["date"])
    X = df.iloc[:-1].drop(columns=["date"], errors="ignore")
    # Align shapes
    X = X.loc[y.index]

    # -------------------------------
    # Train model
    # -------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

       # ---- Store metadata ----
    preds = model.predict(X)
    model.training_mae_ = mean_absolute_error(y, preds)
    model.feature_columns_ = X.columns.tolist()

    model.historical_avg_range_ = df["hl_pct"].mean()
    model.atr_avg_ = df["volatility_10d"].mean()

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{ticker}.pkl")

    print(f"Model trained & saved for {ticker}")

    return model