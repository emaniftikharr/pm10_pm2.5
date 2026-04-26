"""
Align all collected datasets to hourly timestamps for Karachi.
Sources: OpenAQ, AQICN, OpenWeather, GEE (LST, NDVI, land cover).
Output: data/processed/karachi_aligned.parquet
"""

import pandas as pd
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def load_openaq() -> pd.DataFrame:
    path = DATA_RAW_DIR / "openaq_karachi.parquet"
    if not path.exists():
        print("OpenAQ file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.set_index("timestamp").resample("1h").mean(numeric_only=True)
    df.columns = [f"openaq_{c}" for c in df.columns]
    return df


def load_aqicn() -> pd.DataFrame:
    path = DATA_RAW_DIR / "aqicn_karachi.parquet"
    if not path.exists():
        print("AQICN file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.set_index("timestamp")[["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]]
    df = df.resample("1h").mean(numeric_only=True)
    df.columns = [f"aqicn_{c}" for c in df.columns]
    return df


def load_weather() -> pd.DataFrame:
    path = DATA_RAW_DIR / "weather_karachi.parquet"
    if not path.exists():
        print("Weather file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.set_index("timestamp")[[
        "temp_c", "feels_like_c", "humidity_pct", "pressure_hpa",
        "wind_speed_ms", "wind_deg", "clouds_pct", "visibility_m",
    ]]
    df = df.resample("1h").mean(numeric_only=True)
    return df


def load_gee() -> pd.DataFrame:
    path = DATA_RAW_DIR / "gee_karachi.parquet"
    if not path.exists():
        print("GEE file not found, skipping.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")[["LST_C", "NDVI", "urban_fraction"]]
    # Upsample daily GEE data to hourly via forward-fill
    df = df.resample("1h").ffill()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def align() -> pd.DataFrame:
    frames = [load_openaq(), load_aqicn(), load_weather(), load_gee()]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise RuntimeError("No source data found. Run collection scripts first.")

    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")

    df = df.sort_index()
    df = add_time_features(df)

    # Drop rows where ALL pollutant columns are NaN
    pollutant_cols = [c for c in df.columns if any(
        p in c for p in ["AQI", "PM2.5", "PM10", "NO2", "CO"]
    )]
    df = df.dropna(subset=pollutant_cols, how="all")

    out_path = DATA_PROCESSED_DIR / "karachi_aligned.parquet"
    df.to_parquet(out_path)
    print(f"Aligned dataset: {len(df)} rows, {len(df.columns)} columns → {out_path}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"\nColumns:\n{list(df.columns)}")
    return df


if __name__ == "__main__":
    align()
