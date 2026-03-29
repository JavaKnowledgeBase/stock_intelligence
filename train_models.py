"""
train_models.py

Train a stable next-day range model using time-series validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def _prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame[pd.to_numeric(frame["Close"], errors="coerce").notna()].copy()

    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def _candidate_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        ),
    }


def _cross_validated_mae(model, X: pd.DataFrame, y: pd.Series) -> float:
    n_splits = min(5, max(3, len(X) // 60))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []

    for train_idx, test_idx in splitter.split(X):
        candidate = clone(model)
        candidate.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = candidate.predict(X.iloc[test_idx])
        fold_maes.append(mean_absolute_error(y.iloc[test_idx], preds))

    return float(np.mean(fold_maes))


def train_model(ticker: str, df: pd.DataFrame):
    """
    Train a regression model to predict next-day price range.
    """

    frame = _prepare_training_frame(df)
    if len(frame) < 80:
        raise ValueError(f"Not enough clean feature rows to train {ticker}")

    y = frame["hl_pct"].shift(-1).dropna()
    X = frame.iloc[:-1].drop(columns=["date"], errors="ignore")
    X = X.loc[y.index]

    split_index = max(int(len(X) * 0.8), 60)
    split_index = min(split_index, len(X) - 20)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scored_models = []
    for name, model in _candidate_models().items():
        cv_mae = _cross_validated_mae(model, X_train, y_train)
        scored_models.append((cv_mae, name, model))

    scored_models.sort(key=lambda item: item[0])
    best_cv_mae, best_name, best_model = scored_models[0]
    model = clone(best_model)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, X_test["hl_pct"])
    use_baseline = baseline_mae <= test_mae

    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        ranked = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        feature_importance = {
            feature: float(round(score, 6)) for feature, score in ranked
        }

    model.training_mae_ = float(test_mae)
    model.cross_val_mae_ = float(best_cv_mae)
    model.baseline_mae_ = float(baseline_mae)
    model.use_baseline_ = bool(use_baseline)
    model.model_name_ = best_name
    model.feature_columns_ = X.columns.tolist()
    model.historical_avg_range_ = float(frame["hl_pct"].mean())
    model.atr_avg_ = float(frame["volatility_10d"].mean())
    model.prediction_floor_ = float(y_train.quantile(0.1))
    model.prediction_ceiling_ = float(y_train.quantile(0.9))
    model.feature_importance_top_ = feature_importance

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{ticker}.pkl")

    print(f"Model trained & saved for {ticker}")
    print(f"Selected model: {best_name}")
    print(f"Cross-validated MAE: {best_cv_mae:.4f}")
    print(f"Out-of-sample MAE: {test_mae:.4f}")
    print(f"Naive baseline MAE: {baseline_mae:.4f}")
    if use_baseline:
        print("Prediction mode: baseline fallback")
    else:
        print("Prediction mode: trained model")

    return model
