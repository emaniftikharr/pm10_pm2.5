"""
Feature engineering pipeline — reads raw CSV, applies all transforms,
writes feature store to data/processed/feature_store.csv.
Runs after fetch_data.py in the GitHub Actions pipeline.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_FILE      = os.path.join("data", "raw", "karachi_aqi_raw.csv")
FEATURE_STORE = os.path.join("data", "processed", "feature_store.csv")
MODELS_DIR    = "models"

MET_COLS = [
    "temperature_2m", "dew_point_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "cloud_cover", "relative_humidity_2m",
]
POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide",
                  "nitrogen_dioxide", "sulphur_dioxide"]


def load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_FILE):
        print(f"Raw file not found: {RAW_FILE} — run fetch_data.py first.")
        sys.exit(0)
    df = pd.read_csv(RAW_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded raw: {df.shape}")
    return df


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("time").resample("1D").mean(numeric_only=True).reset_index()
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    print(f"Resampled to daily: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    met_present  = [c for c in MET_COLS if c in df.columns]
    poll_present = [c for c in POLLUTANT_COLS if c in df.columns]
    df[met_present]  = df[met_present].interpolate(method="linear", limit_direction="both")
    df[poll_present] = df[poll_present].ffill().bfill()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["pm2_5", "pm10"]:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if "pm2_5" in df.columns:
        df["PM25_roll7"]  = df["pm2_5"].rolling(7,  min_periods=1).mean()
        df["PM25_roll30"] = df["pm2_5"].rolling(30, min_periods=1).mean()
    if "pm10" in df.columns:
        df["PM10_roll7"]  = df["pm10"].rolling(7,  min_periods=1).mean()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    def season(m):
        if m in [12, 1, 2]:  return 1
        if m in [3, 4, 5]:   return 2
        if m in [6, 7, 8]:   return 3
        return 4

    df["season"] = df["month"].apply(season)
    return df


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["pm2_5", "pm10"]:
        if col in df.columns:
            df[f"{col}_change"] = df[col].diff()
    return df


def apply_scalers(df: pd.DataFrame) -> pd.DataFrame:
    std_path = os.path.join(MODELS_DIR, "scaler_standard.pkl")
    mm_path  = os.path.join(MODELS_DIR, "scaler_minmax.pkl")
    met_present = [c for c in MET_COLS if c in df.columns]

    if os.path.exists(std_path) and met_present:
        scaler = joblib.load(std_path)
        try:
            df[met_present] = scaler.transform(df[met_present])
        except Exception as e:
            print(f"  StandardScaler transform skipped: {e}")

    sat_cols = [c for c in ["LST_C", "NDVI"] if c in df.columns]
    if os.path.exists(mm_path) and sat_cols:
        scaler = joblib.load(mm_path)
        try:
            df[sat_cols] = scaler.transform(df[sat_cols])
        except Exception as e:
            print(f"  MinMaxScaler transform skipped: {e}")

    return df


def run():
    print("=" * 50)
    print("Feature Pipeline")
    print(f"Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 50)

    df = load_raw()
    df = resample_daily(df)
    df = handle_missing(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temporal_features(df)
    df = add_change_features(df)
    df = apply_scalers(df)

    # Drop rows with NaN lag columns (first 3 rows)
    lag_cols = [c for c in df.columns if "lag" in c]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(FEATURE_STORE), exist_ok=True)
    df.to_csv(FEATURE_STORE, index=False)
    print(f"\nFeature store saved: {df.shape} -> {FEATURE_STORE}")

    # Log run
    log_path = os.path.join("data", "pipeline_log.csv")
    log = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "rows": len(df),
        "features": len(df.columns),
        "status": "ok",
    }])
    log.to_csv(log_path, mode="a", index=False,
               header=not os.path.exists(log_path))


if __name__ == "__main__":
    run()
