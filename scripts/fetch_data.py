"""
Automated data ingestion — fetches latest air quality + weather data.
Runs hourly via GitHub Actions. Appends new rows to raw CSV.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAT = 24.8607
LON = 67.0011
RAW_FILE = os.path.join("data", "raw", "karachi_aqi_raw.csv")

AQ_URL       = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AQ_VARS      = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
WEATHER_VARS = ("temperature_2m,dew_point_2m,surface_pressure,"
                "wind_speed_10m,wind_direction_10m,precipitation,"
                "cloud_cover,relative_humidity_2m")


def fetch_aq(start: str, end: str) -> pd.DataFrame | None:
    r = requests.get(AQ_URL, params={
        "latitude": LAT, "longitude": LON,
        "hourly": AQ_VARS,
        "start_date": start, "end_date": end,
        "timezone": "Asia/Karachi",
    }, timeout=30)
    if r.status_code != 200:
        print(f"AQ API error {r.status_code}: {r.text[:100]}")
        return None
    return pd.DataFrame(r.json().get("hourly", {}))


def fetch_weather(start: str, end: str, start_dt: datetime) -> pd.DataFrame | None:
    cutoff = datetime.now().date() - timedelta(days=16)
    url = ARCHIVE_URL if start_dt.date() < cutoff else FORECAST_URL
    r = requests.get(url, params={
        "latitude": LAT, "longitude": LON,
        "hourly": WEATHER_VARS,
        "start_date": start, "end_date": end,
        "timezone": "Asia/Karachi",
    }, timeout=60)
    if r.status_code != 200:
        print(f"Weather API error {r.status_code}: {r.text[:100]}")
        return None
    return pd.DataFrame(r.json().get("hourly", {}))


def get_last_timestamp() -> datetime | None:
    if not os.path.exists(RAW_FILE):
        return None
    df = pd.read_csv(RAW_FILE, usecols=["time"], parse_dates=["time"])
    return df["time"].max().to_pydatetime() if not df.empty else None


def append_new_rows(df: pd.DataFrame) -> int:
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    file_exists = os.path.exists(RAW_FILE)

    if file_exists:
        existing = pd.read_csv(RAW_FILE, usecols=["time"], parse_dates=["time"])
        existing_times = set(existing["time"].astype(str))
        df = df[~df["time"].astype(str).isin(existing_times)]

    if df.empty:
        return 0

    df.to_csv(RAW_FILE, mode="a", index=False, header=not file_exists)
    return len(df)


def run():
    now_utc = datetime.now(pytz.UTC)
    last_ts = get_last_timestamp()

    if last_ts:
        last_ts_utc = last_ts.replace(tzinfo=pytz.UTC) if last_ts.tzinfo is None else last_ts
        start_dt = last_ts_utc - timedelta(hours=2)
        if start_dt > now_utc:
            start_dt = now_utc - timedelta(days=2)
    else:
        start_dt = now_utc - timedelta(days=7)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = now_utc.strftime("%Y-%m-%d")

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC] Fetching {start_str} to {end_str}")

    aq = fetch_aq(start_str, end_str)
    wx = fetch_weather(start_str, end_str, start_dt)

    if aq is None or wx is None:
        print("Fetch failed — skipping.")
        sys.exit(1)

    aq["time"] = pd.to_datetime(aq["time"])
    wx["time"] = pd.to_datetime(wx["time"])
    df = pd.merge(aq, wx, on="time", how="inner")

    added = append_new_rows(df)
    print(f"Added {added} new rows to {RAW_FILE}")

    # Write fetch log
    log_path = os.path.join("data", "fetch_log.csv")
    log = pd.DataFrame([{
        "timestamp": now_utc.isoformat(),
        "rows_added": added,
        "start_date": start_str,
        "end_date": end_str,
        "status": "ok",
    }])
    log.to_csv(log_path, mode="a", index=False,
               header=not os.path.exists(log_path))


if __name__ == "__main__":
    run()
