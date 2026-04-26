"""
Phase 3: Data Cleaning & Preprocessing
- Merge all datasets on date
- Handle missing values
- Remove duplicates and outliers
- Normalize features
- Save scalers
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"

for d in [PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Column groups
MET_COLS       = ["temperature_2m", "dew_point_2m", "surface_pressure",
                  "wind_speed_10m", "wind_direction_10m", "precipitation",
                  "cloud_cover", "relative_humidity_2m"]
SATELLITE_COLS = ["LST_C", "NDVI"]
POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide",
                  "nitrogen_dioxide", "sulphur_dioxide"]
AQI_COLS       = ["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_openmeteo() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "karachi_aqi_raw.csv")
    if not os.path.exists(path):
        print("  Open-Meteo file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns={"time": "datetime"})
    df["date"] = df["datetime"].dt.date.astype(str)
    df = df.groupby("date").mean(numeric_only=True).reset_index()
    print(f"  Open-Meteo: {len(df)} daily rows")
    return df


def load_aqicn() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "aqicn_karachi.parquet")
    if not os.path.exists(path):
        print("  AQICN file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
    df = df[[c for c in AQI_COLS + ["date"] if c in df.columns]]
    df = df.groupby("date").mean(numeric_only=True).reset_index()
    df.columns = ["date"] + [f"aqicn_{c}" for c in df.columns if c != "date"]
    print(f"  AQICN: {len(df)} daily rows")
    return df


def load_lst() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "karachi_lst.csv")
    if not os.path.exists(path):
        print("  LST file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date.astype(str)
    print(f"  LST: {len(df)} daily rows")
    return df


def load_ndvi() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "karachi_ndvi.csv")
    if not os.path.exists(path):
        print("  NDVI file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date.astype(str)
    # Forward-fill 16-day composites to daily
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = (df.set_index("date")
            .reindex(full_range.astype(str))
            .ffill()
            .reset_index()
            .rename(columns={"index": "date"}))
    print(f"  NDVI: {len(df)} daily rows (after ffill)")
    return df


# ── 2. MERGE ──────────────────────────────────────────────────────────────────

def merge_all() -> pd.DataFrame:
    print("\n[1] Loading datasets...")
    om   = load_openmeteo()
    aq   = load_aqicn()
    lst  = load_lst()
    ndvi = load_ndvi()

    frames = [f for f in [om, aq, lst, ndvi] if not f.empty]
    if not frames:
        raise RuntimeError("No data files found. Run collection scripts first.")

    print("\n[2] Merging on date...")
    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on="date", how="outer")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Merged shape: {df.shape}")
    return df


# ── 3. MISSING VALUES ─────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Handling missing values...")

    # Flag columns with >30% missing before filling
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.30].index.tolist()
    if high_missing:
        print(f"  WARNING — >30% missing in: {high_missing}")
        for col in high_missing:
            df[f"{col}_high_missing"] = 1

    # Meteorological: linear interpolation
    met_present = [c for c in MET_COLS if c in df.columns]
    df[met_present] = df[met_present].interpolate(method="linear", limit_direction="both")

    # Pollutants + AQI: forward fill then backward fill
    poll_present = [c for c in POLLUTANT_COLS + AQI_COLS if c in df.columns]
    aqicn_cols   = [c for c in df.columns if c.startswith("aqicn_")]
    fill_cols    = list(set(poll_present + aqicn_cols))
    df[fill_cols] = df[fill_cols].ffill().bfill()

    # Satellite: forward fill (already done for NDVI; LST may have cloud gaps)
    sat_present = [c for c in SATELLITE_COLS if c in df.columns]
    df[sat_present] = df[sat_present].ffill().bfill()

    remaining = df.isnull().sum().sum()
    print(f"  Remaining nulls after filling: {remaining}")
    return df


# ── 4. DUPLICATES & OUTLIERS ──────────────────────────────────────────────────

def remove_duplicates_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Removing duplicates and outliers...")

    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    # IQR outlier removal on PM2.5 and PM10 (extreme only: 3×IQR)
    for col in ["pm2_5", "pm10"]:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR
        outliers = (df[col] > upper).sum()
        df = df[df[col] <= upper]
        print(f"  {col}: removed {outliers} extreme outliers (upper bound={upper:.2f})")

    print(f"  Final shape after cleaning: {df.shape}")
    return df.reset_index(drop=True)


# ── 5. NORMALIZE ──────────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Normalizing features...")

    # StandardScaler for meteorological variables
    met_present = [c for c in MET_COLS if c in df.columns]
    if met_present:
        scaler_std = StandardScaler()
        df[met_present] = scaler_std.fit_transform(df[met_present])
        joblib.dump(scaler_std, os.path.join(MODELS_DIR, "scaler_standard.pkl"))
        print(f"  StandardScaler applied to: {met_present}")

    # MinMaxScaler for satellite features
    sat_present = [c for c in SATELLITE_COLS if c in df.columns]
    if sat_present:
        scaler_mm = MinMaxScaler()
        df[sat_present] = scaler_mm.fit_transform(df[sat_present])
        joblib.dump(scaler_mm, os.path.join(MODELS_DIR, "scaler_minmax.pkl"))
        print(f"  MinMaxScaler applied to: {sat_present}")

    return df


# ── 6. SAVE ───────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> str:
    out = os.path.join(PROCESSED_DIR, "karachi_cleaned.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return out


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    print("=" * 55)
    print("Phase 3: Preprocessing Pipeline")
    print("=" * 55)

    df = merge_all()
    df = handle_missing(df)
    df = remove_duplicates_outliers(df)
    df = normalize(df)
    save(df)

    print("\n[Summary]")
    print(f"  Shape      : {df.shape}")
    print(f"  Date range : {df['date'].min()} → {df['date'].max()}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Nulls left : {df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    run_pipeline()
