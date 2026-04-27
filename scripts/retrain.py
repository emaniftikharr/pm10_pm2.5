"""
Daily model retraining — loads latest feature store, retrains best model,
saves versioned .pkl to models/ with timestamp.
Runs at 06:00 UTC daily via GitHub Actions.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_STORE = os.path.join("data", "processed", "feature_store.csv")
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"

TARGETS = ["pm2_5", "pm10"]
EXCLUDE_COLS = ["date"] + TARGETS + [
    "AQI", "PM2.5", "PM10", "aqicn_AQI", "aqicn_PM2.5", "aqicn_PM10",
    "NO2", "CO", "SO2", "O3", "aqicn_NO2", "aqicn_CO", "aqicn_SO2",
]

XGB_PARAMS = {
    "objective":    "reg:squarederror",
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth":    6,
    "random_state": 42,
    "verbosity":    0,
}


def load_feature_store() -> pd.DataFrame:
    if not os.path.exists(FEATURE_STORE):
        raise FileNotFoundError(
            f"Feature store not found: {FEATURE_STORE}\n"
            "Run scripts/feature_pipeline.py first."
        )
    df = pd.read_csv(FEATURE_STORE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded feature store: {df.shape}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    excl = [c for c in EXCLUDE_COLS if c in df.columns]
    return [c for c in df.columns if c not in excl and c != "date"]


def time_split(df: pd.DataFrame):
    n     = len(df)
    train = df.iloc[:int(n * 0.85)]
    test  = df.iloc[int(n * 0.85):]
    return train, test


def train_and_save(df: pd.DataFrame, feature_cols: list[str],
                   target: str, timestamp: str) -> dict:
    if target not in df.columns:
        print(f"  {target} not in data, skipping.")
        return {}

    train, test = time_split(df)
    X_train = train[feature_cols].fillna(0)
    y_train = train[target]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test[target]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"  {target}: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

    # Save versioned model
    versioned_path = os.path.join(MODELS_DIR, f"xgb_{target}_{timestamp}.pkl")
    latest_path    = os.path.join(MODELS_DIR, f"xgb_best_{target}.pkl")
    joblib.dump(model, versioned_path)
    joblib.dump(model, latest_path)
    print(f"  Saved: {versioned_path}")
    print(f"  Updated: {latest_path}")

    return {
        "timestamp": timestamp, "target": target,
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "train_rows": len(train), "test_rows": len(test),
        "model_path": versioned_path,
    }


def cleanup_old_models(keep_last: int = 5):
    """Keep only the last N versioned models per target to save disk space."""
    import glob
    for target in TARGETS:
        pattern = os.path.join(MODELS_DIR, f"xgb_{target}_*.pkl")
        files = sorted(glob.glob(pattern))
        to_delete = files[:-keep_last]
        for f in to_delete:
            if "best" not in f:
                os.remove(f)
                print(f"  Cleaned up: {f}")


def run():
    print("=" * 50)
    print("Daily Model Retraining")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    print(f"Run time: {timestamp} UTC")
    print("=" * 50)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_feature_store()
    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)} columns")

    results = []
    for target in TARGETS:
        print(f"\n[Retraining] {target}")
        result = train_and_save(df, feature_cols, target, timestamp)
        if result:
            results.append(result)

    if results:
        log_df = pd.DataFrame(results)
        log_path = os.path.join(REPORTS_DIR, "retrain_log.csv")
        log_df.to_csv(log_path, mode="a", index=False,
                      header=not os.path.exists(log_path))
        print(f"\nRetrain log saved: {log_path}")

    cleanup_old_models(keep_last=5)
    print("\nRetraining complete.")


if __name__ == "__main__":
    run()
