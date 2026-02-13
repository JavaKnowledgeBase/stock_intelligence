"""
run_pipeline.py
Runs the full pipeline end-to-end.
"""

from config import TICKERS
from download_intraday import download_data as download
from build_features import build_features
from train_models import train_model as train
from predict_next_day import predict_next_day as predict


def run_pipeline():
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")

        # Step 1: Download data
        df = download(ticker)

        if df is None or df.empty:
            print(f"Skipping {ticker}: no data")
            continue

        # Step 2: Build features
        df_features = build_features(df)

        if df_features.empty:
            print(f"Skipping {ticker}: no features")
            continue
        
        import os
        os.makedirs("data/features", exist_ok=True)
        df_features.to_csv(f"data/features/{ticker}.csv", index=False)

        # Step 3: Train model
        model = train(ticker, df_features)

        # Step 4: Predict
        predict(ticker, model, df_features)


if __name__ == "__main__":
    run_pipeline()
