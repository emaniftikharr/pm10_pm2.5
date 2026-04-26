"""
Fetch AQI data from AQICN (waqi.info) for Karachi.
Uses search + direct city feed endpoints (more reliable than bbox).
Extracts: AQI, PM2.5, PM10, NO2, CO, SO2, O3.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN    = os.getenv("AQICN_API_KEY", "")
BASE_URL = "https://api.waqi.info"
OUT_FILE = os.path.join("data", "raw", "aqicn_karachi.parquet")

POLLUTANT_MAP = {
    "aqi":  "AQI",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2":  "NO2",
    "co":   "CO",
    "so2":  "SO2",
    "o3":   "O3",
}

# Known Karachi station slugs + search keywords
KARACHI_FEEDS = ["karachi", "@11371", "@11370", "@13984"]


def fetch_feed(station: str) -> dict:
    resp = requests.get(
        f"{BASE_URL}/feed/{station}/",
        params={"token": TOKEN},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def search_stations(keyword: str) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/search/",
        params={"token": TOKEN, "keyword": keyword},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "ok":
        return data.get("data", [])
    return []


def parse_feed(data: dict) -> dict | None:
    if not data or data.get("status") == "error":
        return None
    feed = data.get("data", {})
    if not feed or feed == "Unknown station":
        return None

    iaqi = feed.get("iaqi", {})
    city = feed.get("city", {})

    row = {
        "timestamp":    pd.to_datetime(feed.get("time", {}).get("iso"), utc=True),
        "station_name": city.get("name", ""),
        "lat":          city.get("geo", [None, None])[0],
        "lon":          city.get("geo", [None, None])[1],
        "AQI":          feed.get("aqi"),
    }
    for key, col in POLLUTANT_MAP.items():
        if key != "aqi":
            row[col] = iaqi.get(key, {}).get("v", None)

    return row


def collect() -> pd.DataFrame:
    rows = []

    # 1. Search for all Karachi stations
    print("Searching for Karachi stations...")
    results = search_stations("Karachi")
    print(f"Found {len(results)} stations via search")

    for station in results:
        uid = station.get("uid")
        if uid:
            data = fetch_feed(f"@{uid}")
            row = parse_feed(data)
            if row:
                rows.append(row)
                print(f"  Got data from: {row['station_name']}")
            time.sleep(0.3)

    # 2. Also try known slugs as fallback
    for slug in KARACHI_FEEDS:
        data = fetch_feed(slug)
        row = parse_feed(data)
        if row and not any(r["station_name"] == row["station_name"] for r in rows):
            rows.append(row)
            print(f"  Got data from slug '{slug}': {row['station_name']}")
        time.sleep(0.3)

    if not rows:
        print("No AQICN data retrieved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\nAQICN: {len(df)} stations saved → {OUT_FILE}")
    print(df[["station_name", "AQI", "PM2.5", "PM10"]].to_string(index=False))
    return df


if __name__ == "__main__":
    collect()
