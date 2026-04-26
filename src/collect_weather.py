"""
Fetch historical weather data from OpenWeatherMap for Karachi.
Extracts: temperature, humidity, wind speed, pressure, weather condition.
Uses the One Call API 3.0 (history endpoint).
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import DATA_RAW_DIR

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011


def fetch_day(dt_unix: int) -> list[dict]:
    resp = requests.get(
        BASE_URL,
        params={
            "lat": KARACHI_LAT,
            "lon": KARACHI_LON,
            "dt": dt_unix,
            "appid": API_KEY,
            "units": "metric",
        },
        timeout=30,
    )
    if resp.status_code == 429:
        time.sleep(60)
        return fetch_day(dt_unix)
    resp.raise_for_status()
    return resp.json().get("data", [])


def parse_hourly(hourly: list[dict]) -> list[dict]:
    rows = []
    for h in hourly:
        rows.append({
            "timestamp": pd.to_datetime(h["dt"], unit="s", utc=True),
            "temp_c": h.get("temp"),
            "feels_like_c": h.get("feels_like"),
            "humidity_pct": h.get("humidity"),
            "pressure_hpa": h.get("pressure"),
            "wind_speed_ms": h.get("wind_speed"),
            "wind_deg": h.get("wind_deg"),
            "clouds_pct": h.get("clouds"),
            "visibility_m": h.get("visibility"),
            "weather": h.get("weather", [{}])[0].get("description", ""),
            "lat": KARACHI_LAT,
            "lon": KARACHI_LON,
        })
    return rows


def collect(days_back: int = 365) -> pd.DataFrame:
    all_rows = []
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    for i in range(days_back):
        target_day = today - timedelta(days=i + 1)
        dt_unix = int(target_day.timestamp())
        try:
            hourly = fetch_day(dt_unix)
            all_rows.extend(parse_hourly(hourly))
        except Exception as e:
            print(f"  Failed {target_day.date()}: {e}")
        time.sleep(0.2)

        if (i + 1) % 30 == 0:
            print(f"  Progress: {i + 1}/{days_back} days fetched")

    if not all_rows:
        print("No weather data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)
    out_path = DATA_RAW_DIR / "weather_karachi.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Weather: {len(df)} rows saved → {out_path}")
    return df


if __name__ == "__main__":
    collect(days_back=365)
