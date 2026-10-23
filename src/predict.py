"""
predict.py - Machine-learning forecast of future parking occupancy.

Target  : average occupancy % in a future hour.
Features: hour, day of week, weekend flag,
          lag_24  = occupancy at the same hour yesterday,
          lag_168 = occupancy at the same hour last week.
Because both lags are at least 24 h old, we can forecast up to 24 h ahead
using only data we already have.

Run:
    python -m src.predict --source synthetic     # train, evaluate, save, forecast
"""
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DB_PATH, FORECAST_HOURS, MIN_HISTORY_DAYS, PRED_MODEL_PATH
from src.analysis import clean_records
from src.database import fetch_records

FEATURES = ["hour", "dayofweek", "is_weekend", "lag_24", "lag_168"]


class InsufficientDataError(Exception):
    pass


def prepare_hourly(df):
    """Convert raw observations into a regular hourly series of occupancy %."""
    df, _ = clean_records(df)
    series = df.set_index("timestamp")["occupancy_pct"].resample("h").mean()
    total_slots = int(df["total_slots"].iloc[-1])
    return series, total_slots


def make_features(series):
    frame = pd.DataFrame({"occupancy_pct": series})
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
    frame["lag_24"] = series.shift(24)       # works because the index is regular hourly
    frame["lag_168"] = series.shift(168)
    return frame.dropna()


def _metrics(y_true, y_pred):
    return {"MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2)}


def train_and_evaluate(series, test_fraction=0.2):
    """Time-ordered split, baselines vs models. Returns a model bundle dict."""
    span_days = (series.index.max() - series.index.min()).days if len(series) else 0
    feat = make_features(series)
    if span_days < MIN_HISTORY_DAYS or len(feat) < 24 * 7:
        raise InsufficientDataError(
            f"Only {span_days} days of history ({len(feat)} usable hourly rows). "
            f"Need at least {MIN_HISTORY_DAYS} days. Collect more real data, or test the "
            "pipeline with:  python -m src.generate_synthetic"
        )

    # Train on the past, test on the most recent part. A random split would let
    # the model peek at the future (data leakage) and give over-optimistic scores.
    split = int(len(feat) * (1 - test_fraction))
    train, test = feat.iloc[:split], feat.iloc[split:]
    X_train, y_train = train[FEATURES], train["occupancy_pct"]
    X_test, y_test = test[FEATURES], test["occupancy_pct"]

    results = {
        "Baseline: same hour yesterday": _metrics(y_test, test["lag_24"]),
        "Baseline: same hour last week": _metrics(y_test, test["lag_168"]),
    }
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, 100)
        results[name] = _metrics(y_test, pred)

    best_name = min(candidates, key=lambda n: results[n]["MAE"])
    best_baseline = min((n for n in results if n.startswith("Baseline")),
                        key=lambda n: results[n]["MAE"])
    beats_baseline = results[best_name]["MAE"] < results[best_baseline]["MAE"]

    # Refit the chosen model on ALL data before using it for real forecasts.
    final_model = candidates[best_name].fit(feat[FEATURES], feat["occupancy_pct"])

    return {
        "model": final_model,
        "model_name": best_name,
        "features": FEATURES,
        "metrics": results,
        "best_baseline": best_baseline,
        "beats_baseline": beats_baseline,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_period": (str(test.index.min()), str(test.index.max())),
        "trained_until": str(series.index.max()),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def forecast(bundle, series, total_slots, hours=FORECAST_HOURS):
    """Predict occupancy for the next `hours` hours after the last observation."""
    if not 1 <= hours <= 24:
        raise ValueError("Forecast horizon must be 1-24 hours (lag_24 must be known).")
    hour_means = series.groupby(series.index.hour).mean()     # fallback for gaps

    future = pd.date_range(series.index.max() + pd.Timedelta(hours=1), periods=hours, freq="h")
    rows = []
    for t in future:
        lag24 = series.get(t - pd.Timedelta(hours=24), np.nan)
        lag168 = series.get(t - pd.Timedelta(hours=168), np.nan)
        if pd.isna(lag24):
            lag24 = hour_means.get(t.hour, series.mean())
        if pd.isna(lag168):
            lag168 = hour_means.get(t.hour, series.mean())
        rows.append({"hour": t.hour, "dayofweek": t.dayofweek,
                     "is_weekend": int(t.dayofweek >= 5),
                     "lag_24": lag24, "lag_168": lag168})

    X = pd.DataFrame(rows, index=future)[bundle["features"]]
    pct = np.clip(bundle["model"].predict(X), 0, 100)          # never below 0 or above 100
    occupied = np.clip(np.rint(pct / 100 * total_slots), 0, total_slots).astype(int)
    return pd.DataFrame({
        "predicted_occupancy_pct": pct.round(1),
        "predicted_occupied": occupied,
        "predicted_available": total_slots - occupied,
    }, index=future.rename("timestamp"))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the occupancy model.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "video", "live"])
    parser.add_argument("--hours", type=int, default=FORECAST_HOURS)
    args = parser.parse_args()

    df = fetch_records(DB_PATH, args.source)
    if df.empty:
        raise SystemExit(f"No '{args.source}' records found.")
    series, total_slots = prepare_hourly(df)

    if args.source == "synthetic":
        print("*** Training on SYNTHETIC data: this tests the pipeline, it does NOT "
              "predict a real car park. ***\n")
    try:
        bundle = train_and_evaluate(series)
    except InsufficientDataError as e:
        raise SystemExit(str(e))

    bundle["source"] = args.source
    bundle["total_slots"] = total_slots

    print(f"Train rows: {bundle['train_rows']}, test rows: {bundle['test_rows']} "
          f"(test period {bundle['test_period'][0]} -> {bundle['test_period'][1]})")
    print("Test error, in percentage points of occupancy (lower is better):")
    for name, m in bundle["metrics"].items():
        print(f"  {name:<32} MAE {m['MAE']:>6}   RMSE {m['RMSE']:>6}")
    print(f"Chosen model: {bundle['model_name']}")
    if not bundle["beats_baseline"]:
        print("WARNING: the ML model does not beat the simple baseline on this data.")

    PRED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, PRED_MODEL_PATH)
    print(f"Saved model to {PRED_MODEL_PATH}\n")

    fc = forecast(bundle, series, total_slots, args.hours)
    print(f"Forecast for the next {args.hours} h ({total_slots} slots):")
    print(fc.to_string())


if __name__ == "__main__":
    main()
