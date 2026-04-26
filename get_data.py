"""
Run this from the project root:
    python get_data.py

Saves to: data/raw/karachi_aqi_raw.csv
No API keys needed.
"""

import os
import re
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

LAT = 24.8607
LON = 67.0011
OUT_FILE = os.path.join("data", "raw", "karachi_aqi_raw.csv")

AQ_URL      = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL= "https://api.open-meteo.com/v1/forecast"

AQ_VARS      = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
WEATHER_VARS = "temperature_2m,dew_point_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,relative_humidity_2m"


def fetch_aq(start, end):
    r = requests.get(AQ_URL, params={
        "latitude": LAT, "longitude": LON,
        "hourly": AQ_VARS,
        "start_date": start, "end_date": end,
        "timezone": "Asia/Karachi"
    }, timeout=30)
    if r.status_code != 200:
        print(f"  AQ API error: {r.status_code}")
        return None
    return pd.DataFrame(r.json().get("hourly", {}))


def fetch_weather(start, end, start_dt):
    cutoff = datetime.now().date() - timedelta(days=16)
    url = ARCHIVE_URL if start_dt.date() < cutoff else FORECAST_URL
    r = requests.get(url, params={
        "latitude": LAT, "longitude": LON,
        "hourly": WEATHER_VARS,
        "start_date": start, "end_date": end,
        "timezone": "Asia/Karachi"
    }, timeout=60)
    if r.status_code != 200:
        print(f"  Weather API error: {r.status_code}")
        return None
    return pd.DataFrame(r.json().get("hourly", {}))


def save(df):
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    file_exists = os.path.exists(OUT_FILE)
    df.to_csv(OUT_FILE, mode="a", index=False, header=not file_exists)


def main():
    print("=" * 50)
    print("Karachi Air Quality Data Collection")
    print(f"Output: {OUT_FILE}")
    print("=" * 50)

    start_dt = datetime(2025, 1, 1)
    end_dt   = datetime.now()
    delta    = timedelta(days=7)
    current  = start_dt
    batch    = 0
    total_rows = 0

    while current <= end_dt:
        batch += 1
        s = current.strftime("%Y-%m-%d")
        e = (current + delta).strftime("%Y-%m-%d")
        print(f"\nBatch {batch}: {s} → {e}")

        try:
            aq = fetch_aq(s, e)
            wx = fetch_weather(s, e, current)

            if aq is not None and wx is not None and not aq.empty and not wx.empty:
                aq["time"] = pd.to_datetime(aq["time"])
                wx["time"] = pd.to_datetime(wx["time"])
                df = pd.merge(aq, wx, on="time", how="inner")
                save(df)
                total_rows += len(df)
                print(f"  Saved {len(df)} rows  (total: {total_rows})")
            else:
                print("  Skipped — empty response")

        except Exception as ex:
            print(f"  Error: {ex}")

        time.sleep(1)
        current += delta

    print(f"\nDone! {total_rows} rows saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
