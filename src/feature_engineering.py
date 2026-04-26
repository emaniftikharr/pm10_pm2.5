"""
Phase 3: Feature Engineering
- Lag features for PM2.5, PM10, AQI
- Rolling averages (7-day AQI, 30-day LST)
- Temporal features (month, season, day_of_week, is_weekend)
- AQI change rate
- Train / validation / test split (time-aware)
"""

import os
import joblib
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"

# Column name mapping — handles both raw and aqicn-prefixed names
PM25_COL = None   # resolved at runtime
PM10_COL = None
AQI_COL  = None


def resolve_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    """Find the actual PM2.5, PM10, AQI column names in the dataframe."""
    candidates = {
        "pm25": ["pm2_5", "PM2.5", "aqicn_PM2.5", "pm25"],
        "pm10": ["pm10", "PM10", "aqicn_PM10"],
        "aqi":  ["AQI",  "aqicn_AQI"],
    }
    found = {}
    for key, options in candidates.items():
        for col in options:
            if col in df.columns:
                found[key] = col
                break
        if key not in found:
            found[key] = None
    return found["pm25"], found["pm10"], found["aqi"]


# ── 1. LAG FEATURES ───────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, pm25: str, pm10: str, aqi: str) -> pd.DataFrame:
    print("[1] Creating lag features...")
    for lag in [1, 2, 3]:
        if pm25:
            df[f"PM25_lag{lag}"] = df[pm25].shift(lag)
        if pm10:
            df[f"PM10_lag{lag}"] = df[pm10].shift(lag)
        if aqi:
            df[f"AQI_lag{lag}"] = df[aqi].shift(lag)
    print(f"    Added lag 1/2/3 for PM2.5, PM10, AQI")
    return df


# ── 2. ROLLING AVERAGES ───────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, pm25: str, pm10: str, aqi: str) -> pd.DataFrame:
    print("[2] Computing rolling averages...")
    if aqi:
        df["AQI_roll7"]  = df[aqi].rolling(7,  min_periods=1).mean()
        df["AQI_roll30"] = df[aqi].rolling(30, min_periods=1).mean()
        print("    AQI 7-day and 30-day rolling mean")
    if pm25:
        df["PM25_roll7"] = df[pm25].rolling(7, min_periods=1).mean()
        print("    PM2.5 7-day rolling mean")
    if pm10:
        df["PM10_roll7"] = df[pm10].rolling(7, min_periods=1).mean()
        print("    PM10 7-day rolling mean")
    if "LST_C" in df.columns:
        df["LST_roll30"] = df["LST_C"].rolling(30, min_periods=1).mean()
        print("    LST 30-day rolling mean")
    return df


# ── 3. TEMPORAL FEATURES ──────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3] Extracting temporal features...")
    df["date"] = pd.to_datetime(df["date"])
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek      # 0=Monday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    def get_season(month):
        if month in [12, 1, 2]:  return 1   # Winter
        elif month in [3, 4, 5]: return 2   # Spring
        elif month in [6, 7, 8]: return 3   # Summer
        else:                    return 4   # Autumn

    df["season"] = df["month"].apply(get_season)
    print("    month, season, day_of_week, is_weekend, day_of_year")
    return df


# ── 4. AQI CHANGE RATE ────────────────────────────────────────────────────────

def add_change_rate(df: pd.DataFrame, pm25: str, pm10: str, aqi: str) -> pd.DataFrame:
    print("[4] Computing change rates...")
    if aqi:
        df["AQI_change"]  = df[aqi].diff()
        print("    AQI_change (day-over-day diff)")
    if pm25:
        df["PM25_change"] = df[pm25].diff()
        print("    PM25_change")
    if pm10:
        df["PM10_change"] = df[pm10].diff()
        print("    PM10_change")
    return df


# ── 5. TRAIN / VAL / TEST SPLIT ───────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[5] Splitting data (time-aware, no shuffle)...")
    df["date"] = pd.to_datetime(df["date"])

    # Determine available date range
    min_year = df["date"].dt.year.min()
    max_year = df["date"].dt.year.max()
    print(f"    Available range: {min_year} → {max_year}")

    if max_year >= 2024:
        train = df[df["date"].dt.year <= 2022]
        val   = df[df["date"].dt.year == 2023]
        test  = df[df["date"].dt.year == 2024]
    elif max_year >= 2025:
        train = df[df["date"] < "2025-10-01"]
        val   = df[(df["date"] >= "2025-10-01") & (df["date"] < "2026-01-01")]
        test  = df[df["date"] >= "2026-01-01"]
    else:
        # Fallback: 70/15/15 time split
        n     = len(df)
        train = df.iloc[:int(n * 0.70)]
        val   = df.iloc[int(n * 0.70):int(n * 0.85)]
        test  = df.iloc[int(n * 0.85):]

    print(f"    Train : {len(train)} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"    Val   : {len(val)} rows  ({val['date'].min().date()} → {val['date'].max().date()})" if not val.empty else "    Val   : 0 rows")
    print(f"    Test  : {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})" if not test.empty else "    Test  : 0 rows")
    return train, val, test


# ── 6. SAVE ───────────────────────────────────────────────────────────────────

def save_splits(df: pd.DataFrame, train: pd.DataFrame,
                val: pd.DataFrame, test: pd.DataFrame) -> None:
    df.to_csv(os.path.join(PROCESSED_DIR, "feature_matrix.csv"), index=False)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"),   index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    # Save column list for modeling phase
    joblib.dump(list(df.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    print(f"\n  feature_matrix.csv  → {df.shape}")
    print(f"  train.csv           → {train.shape}")
    print(f"  val.csv             → {val.shape}")
    print(f"  test.csv            → {test.shape}")
    print(f"  feature_columns.pkl → {len(df.columns)} columns saved")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    print("=" * 55)
    print("Phase 3: Feature Engineering")
    print("=" * 55)

    path = os.path.join(PROCESSED_DIR, "karachi_cleaned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run src/preprocess.py first to generate karachi_cleaned.csv")

    df = pd.read_csv(path)
    print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} cols")

    pm25, pm10, aqi = resolve_cols(df)
    print(f"Detected columns — PM2.5: {pm25} | PM10: {pm10} | AQI: {aqi}\n")

    df = add_lag_features(df, pm25, pm10, aqi)
    df = add_rolling_features(df, pm25, pm10, aqi)
    df = add_temporal_features(df)
    df = add_change_rate(df, pm25, pm10, aqi)

    # Drop rows with NaN from lag/diff (first 3 rows)
    df = df.dropna(subset=[c for c in df.columns if "lag" in c or "change" in c])
    df = df.reset_index(drop=True)

    train, val, test = split_data(df)

    print("\n[6] Saving outputs...")
    save_splits(df, train, val, test)

    print(f"\nDone! Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    run_pipeline()
