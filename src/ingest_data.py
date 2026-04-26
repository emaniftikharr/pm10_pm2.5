"""
Raw data ingestion for Karachi air quality.
Sources: Open-Meteo Air Quality API + Open-Meteo Weather API (no API keys required).
Saves to: data/karachi_aqi_raw.csv
"""

import os
import re
import time
import threading
import requests
import schedule
import pytz
import pandas as pd
from datetime import datetime, timedelta

from config import DATA_RAW_DIR

LAT = 24.8607
LON = 67.0011
RAW_OUTPUT_FILE = DATA_RAW_DIR / "karachi_aqi_raw.csv"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_VARIABLES = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide"
WEATHER_VARIABLES = (
    "temperature_2m,dew_point_2m,surface_pressure,"
    "wind_speed_10m,wind_direction_10m,precipitation,"
    "cloud_cover,relative_humidity_2m"
)


def get_forecast_date_range() -> tuple[datetime, datetime]:
    """Probe the forecast API to find its allowed date range."""
    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": 0,
                "longitude": 0,
                "hourly": "temperature_2m",
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
            },
            timeout=30,
        )
        if "out of allowed range" in resp.text:
            match = re.search(r"from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", resp.text)
            if match:
                min_d = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                max_d = datetime.strptime(match.group(2), "%Y-%m-%d").replace(tzinfo=pytz.UTC)
                print(f"📅 Forecast API range: {min_d.date()} → {max_d.date()}")
                return min_d, max_d
    except Exception as e:
        print(f"⚠️ Error checking forecast range: {e}")

    now = datetime.now(pytz.UTC)
    return now - timedelta(days=90), now + timedelta(days=14)


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch raw air quality data from Open-Meteo."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": AQ_VARIABLES,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    print(f"📡 AQ API: {start_date} → {end_date}")
    resp = requests.get(AQ_URL, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"❌ AQ API Error {resp.status_code}: {resp.text[:200]}")
        return None
    df = pd.DataFrame(resp.json().get("hourly", {}))
    print(f"   AQ shape: {df.shape}")
    return df


def fetch_weather(start_date: str, end_date: str, start_date_obj: datetime) -> pd.DataFrame | None:
    """Fetch weather data, choosing archive vs forecast endpoint by date."""
    cutoff = datetime.now().date() - timedelta(days=16)
    start_d = start_date_obj.date() if isinstance(start_date_obj, datetime) else start_date_obj

    url = ARCHIVE_URL if start_d < cutoff else FORECAST_URL
    label = "archive" if start_d < cutoff else "forecast"
    print(f"📡 Weather ({label}): {start_date} → {end_date}")

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": WEATHER_VARIABLES,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"❌ Weather API Error {resp.status_code}: {resp.text[:200]}")
        return None
    df = pd.DataFrame(resp.json().get("hourly", {}))
    print(f"   Weather shape: {df.shape}")
    return df


def merge_sources(aq: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    aq["time"] = pd.to_datetime(aq["time"])
    weather["time"] = pd.to_datetime(weather["time"])
    df = pd.merge(aq, weather, on="time", how="inner")
    return df


def save_raw(df: pd.DataFrame, is_realtime: bool = False) -> None:
    """Append-safe CSV save."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    file_exists = RAW_OUTPUT_FILE.exists()

    if is_realtime and file_exists:
        existing = pd.read_csv(RAW_OUTPUT_FILE, usecols=["time"], parse_dates=["time"])
        latest = pd.to_datetime(df["time"].iloc[0])
        if latest in existing["time"].values:
            print(f"⚠️ {latest} already exists, skipping.")
            return
        df.to_csv(RAW_OUTPUT_FILE, mode="a", index=False, header=False)
        print(f"✅ Appended 1 row → {RAW_OUTPUT_FILE}")
    elif not file_exists:
        df.to_csv(RAW_OUTPUT_FILE, index=False)
        print(f"✅ Created {RAW_OUTPUT_FILE} ({len(df)} rows)")
    else:
        df.to_csv(RAW_OUTPUT_FILE, mode="a", index=False, header=False)
        print(f"✅ Appended {len(df)} rows → {RAW_OUTPUT_FILE}")


def fetch_realtime() -> None:
    """Fetch the latest hour of data and append to CSV."""
    utc = pytz.UTC
    karachi = pytz.timezone("Asia/Karachi")
    now_utc = datetime.now(utc)
    now_pk = now_utc.astimezone(karachi)
    print(f"\n🔄 [{now_pk.strftime('%Y-%m-%d %H:%M:%S')} PKT] Real-time fetch...")

    two_hours_ago = now_utc - timedelta(hours=2)
    start_str = two_hours_ago.strftime("%Y-%m-%d")
    end_str = now_utc.strftime("%Y-%m-%d")

    try:
        aq = fetch_air_quality(start_str, end_str)
        if aq is None:
            return
        weather = fetch_weather(start_str, end_str, two_hours_ago)
        if weather is None:
            return

        df = merge_sources(aq, weather)
        if df.empty:
            print("⚠️ No merged data for real-time update.")
            return

        df = df.sort_values("time").tail(1)
        save_raw(df, is_realtime=True)

    except Exception as e:
        print(f"❌ Real-time fetch error: {e}")


def fetch_historical(start_year: int = 2025, days_per_batch: int = 7) -> None:
    """Backfill historical data in weekly batches."""
    utc = pytz.UTC
    start_date = datetime(start_year, 1, 1, tzinfo=utc)
    end_date = datetime.now(utc)
    delta = timedelta(days=days_per_batch)
    current = start_date
    batch = 0

    print(f"🚀 Historical backfill: {start_date.date()} → {end_date.date()}")

    while current <= end_date:
        batch += 1
        chunk_start = current.strftime("%Y-%m-%d")
        chunk_end = (current + delta).strftime("%Y-%m-%d")
        print(f"\n📦 Batch {batch}: {chunk_start} → {chunk_end}")

        try:
            aq = fetch_air_quality(chunk_start, chunk_end)
            if aq is None:
                current += delta
                continue

            weather = fetch_weather(chunk_start, chunk_end, current)
            if weather is None:
                current += delta
                continue

            df = merge_sources(aq, weather)
            if not df.empty:
                save_raw(df, is_realtime=False)
                print(f"   ✅ {len(df)} rows saved")
            else:
                print(f"   ⚠️ Empty merge for this batch")

        except Exception as e:
            print(f"❌ Batch error: {e}")

        time.sleep(1)
        current += delta

    print(f"\n✅ Historical ingestion complete → {RAW_OUTPUT_FILE}")


def _run_scheduler() -> None:
    while True:
        schedule.run_pending()
        time.sleep(60)


def start_realtime_ingestion() -> threading.Thread:
    """Schedule hourly real-time fetches in a background thread."""
    schedule.every().hour.at(":05").do(fetch_realtime)
    thread = threading.Thread(target=_run_scheduler, daemon=True)
    thread.start()

    print("🔄 Running initial real-time fetch...")
    fetch_realtime()
    print("🎯 Real-time ingestion active — updates every hour at :05")
    return thread


def main() -> None:
    print(f"📍 Karachi ({LAT}, {LON})")
    print(f"💾 Output: {RAW_OUTPUT_FILE}")

    fetch_historical(start_year=2025)
    start_realtime_ingestion()

    print("\n🔄 Background ingestion running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")


if __name__ == "__main__":
    main()
