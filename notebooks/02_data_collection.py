# Phase 2: Data Collection — Karachi Air Quality
# Uses Open-Meteo APIs (no API keys required)
# Output: data/karachi_aqi_raw.csv

import sys
sys.path.append('../')

from src.ingest_data import fetch_historical, start_realtime_ingestion, RAW_OUTPUT_FILE
import pandas as pd


def run_historical_only():
    """Backfill historical data without starting real-time scheduler."""
    fetch_historical(start_year=2025)


def run_full_pipeline():
    """Backfill history then start hourly real-time updates."""
    import time
    fetch_historical(start_year=2025)
    start_realtime_ingestion()
    print("🔄 Real-time ingestion running in background. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Stopped.")


def preview_data():
    """Quick look at the collected raw data."""
    if not RAW_OUTPUT_FILE.exists():
        print("No raw data yet. Run run_historical_only() first.")
        return None
    df = pd.read_csv(RAW_OUTPUT_FILE, parse_dates=["time"])
    print(f"Shape      : {df.shape}")
    print(f"Date range : {df['time'].min()} → {df['time'].max()}")
    print(f"\nMissing (%):\n{(df.isnull().mean() * 100).round(2).to_string()}")
    print(f"\nSample:\n{df.head()}")
    return df


if __name__ == "__main__":
    run_full_pipeline()
