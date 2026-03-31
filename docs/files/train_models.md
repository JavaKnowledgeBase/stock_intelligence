# File Reference: train_models.py

## Purpose

Trains machine learning models to predict the next day's high-low percentage range (intraday volatility) for each ticker. This is an offline script — not called at runtime by the dashboard or API. Run it locally after downloading fresh data, then upload the resulting `.pkl` files to R2.

---

## Libraries Used

| Library | Why |
|---|---|
| `scikit-learn` | RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, TimeSeriesSplit, mean_absolute_error |
| `joblib` | Serialise trained model + metadata dict to `.pkl` |
| `pandas` | DataFrame operations, numeric coercion |
| `numpy` | Inf handling, percentile computation |
| `pathlib` | Find feature CSVs in `data/features/` |
| `config` | DATA_DIR path |

---

## Workflow

```
data/features/{ticker}.csv   ← pre-built by build_features.py
         │
         ▼
load_frame(path)              ← load + validate
         │
         ▼
prepare_frame(df)
  ├── to_numeric, inf→NaN, dropna
  ├── target = hl_pct.shift(-1)  ← NEXT day's range
  └── features = all cols except date
         │
         ▼
train_model(df, ticker)
  ├── 80/20 time-ordered train/test split
  ├── TimeSeriesSplit cross-validation (3-5 folds)
  ├── compete: RandomForest vs ExtraTrees vs GradientBoosting
  ├── select lowest CV MAE
  ├── train winner on full train set
  ├── compare test MAE vs naive baseline
  └── use_baseline_ = True if model loses to naive
         │
         ▼
save to models/{ticker}.pkl   ← joblib.dump(metadata_dict)
```

---

## Target Variable

```python
target = df["hl_pct"].shift(-1)
```

`hl_pct` = (High − Low) / Close × 100 for a given day. Shifting by -1 means the target for row t is the actual intraday range of day t+1. This is what the model is predicting: "given today's features, what will tomorrow's volatility range be?"

The last row (which has NaN as its target after the shift) is dropped before training.

---

## Train/Test Split

```python
split_idx = int(len(df) * 0.80)
X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_test  = y.iloc[split_idx:]
```

**Why ordered split?** Financial time series have temporal dependencies. Random splitting would allow the model to learn from "future" data points, producing falsely optimistic validation scores. The strict 80% past / 20% future split reflects real deployment conditions.

Minimum: 60 training rows. If fewer, the ticker is skipped.

---

## Model Candidates

### RandomForestRegressor
```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
```
Ensemble of 500 decision trees with bootstrap sampling and random feature subsets. `max_depth=8` prevents overfitting on noisy financial data. `min_samples_leaf=4` ensures each leaf has enough data points.

### ExtraTreesRegressor
```python
ExtraTreesRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
```
Like RandomForest but also randomises split thresholds, further reducing variance. Slightly deeper trees (`max_depth=10`) allowed because the extra randomisation provides implicit regularisation.

### GradientBoostingRegressor
```python
GradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.03,
    n_estimators=300,
    max_depth=2,
    subsample=0.8,
    random_state=42
)
```
Sequential tree building where each tree corrects residuals from the previous. `loss="absolute_error"` makes it robust to outlier days (earnings, macro shocks). Low `learning_rate=0.03` with `n_estimators=300` for stable convergence. Shallow trees (`max_depth=2`) prevent overfitting to specific patterns.

---

## Cross-Validation

```python
n_folds = 5 if len(X_train) >= 200 else 3
tscv = TimeSeriesSplit(n_splits=n_folds)
```

`TimeSeriesSplit` ensures each fold's validation data comes strictly after its training data — preserving temporal order. The fold that produces the best (lowest) mean MAE across splits is the winner.

---

## Baseline Comparison

The naive baseline predicts `historical_avg_range_` (mean of all training targets) for every row:

```python
baseline_preds = np.full(len(y_test), y_train.mean())
baseline_mae   = mean_absolute_error(y_test, baseline_preds)
model_mae      = mean_absolute_error(y_test, model.predict(X_test))

use_baseline   = model_mae >= baseline_mae
```

If the trained model is no better than "always predict the average", `use_baseline_=True` is set and the API will use `historical_avg_range_` as the prediction instead.

---

## Model Metadata (saved in `.pkl`)

```python
{
    "model":                 sklearn_estimator,
    "model_name_":           "RandomForestRegressor",
    "feature_columns_":      [list of feature names],
    "training_mae_":         float,   # MAE on training set
    "cross_val_mae_":        float,   # mean MAE from TimeSeriesSplit
    "baseline_mae_":         float,   # MAE of naive baseline on test set
    "use_baseline_":         bool,    # True = model lost to naive
    "historical_avg_range_": float,   # mean(y_train) — used if use_baseline_
    "atr_avg_":              float,   # mean of atr_14 column
    "prediction_floor_":     float,   # 10th percentile of y_train
    "prediction_ceiling_":   float,   # 90th percentile of y_train
    "feature_importance_top_": list,  # top 10 features if model has feature_importances_
}
```

---

## Output

Files are saved to `models/{ticker}.pkl`. Loading:

```python
import joblib
data = joblib.load("models/AAPL.pkl")
model = data["model"]
prediction = model.predict(X_new)
```

---

## Usage Note

This script is not imported by the dashboard. It is run offline as part of the data pipeline:

```bash
python download_intraday.py    # download fresh 2-year OHLCV
# build_features.py is called internally or separately
python train_models.py         # retrain all models
# then upload models/ to R2 if deploying remotely
```
