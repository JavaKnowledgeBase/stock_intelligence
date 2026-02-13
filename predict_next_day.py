"""
predict_next_day.py

Generates next-day predictions using a trained model.
"""

import pandas as pd


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

    # Use the most recent row of features
    X_latest = df.iloc[[-1]].drop(columns=["date"])

    prediction = model.predict(X_latest)[0]

    print(
        f"Next-day predicted high-low range for {ticker}: "
        f"{prediction:.2f}%"
    )

    return prediction
