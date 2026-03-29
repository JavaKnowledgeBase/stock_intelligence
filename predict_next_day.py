"""
predict_next_day.py

Generates next-day predictions using a trained model.
"""

import pandas as pd


def _should_use_baseline(model) -> bool:
    explicit_flag = getattr(model, "use_baseline_", None)
    if explicit_flag is not None:
        return bool(explicit_flag)

    training_mae = getattr(model, "training_mae_", None)
    baseline_mae = getattr(model, "baseline_mae_", None)
    if training_mae is None or baseline_mae is None:
        return False

    return float(baseline_mae) <= float(training_mae)


def predict_next_day(ticker: str, model, df: pd.DataFrame):
    """
    Predict next-day high-low percentage range.

    Parameters
    ----------
    ticker : str
        Stock symbol (for logging)
    model : trained ML model
    df : pd.DataFrame
        Feature-engineered DataFrame

    Returns
    -------
    float
        Predicted next-day high-low percentage
    """

    frame = df.copy()
    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna().reset_index(drop=True)
    X_latest = frame.iloc[[-1]].drop(columns=["date"], errors="ignore")
    X_latest = X_latest.reindex(columns=model.feature_columns_, fill_value=0)

    if _should_use_baseline(model):
        prediction = float(frame.iloc[-1]["hl_pct"])
    else:
        prediction = float(model.predict(X_latest)[0])

    prediction_floor = getattr(model, "prediction_floor_", 0.5)
    prediction_ceiling = getattr(model, "prediction_ceiling_", 5.0)
    prediction = max(min(prediction, prediction_ceiling), prediction_floor)

    print(
        f"Next-day predicted high-low range for {ticker}: "
        f"{prediction:.2f}%"
    )

    return prediction
