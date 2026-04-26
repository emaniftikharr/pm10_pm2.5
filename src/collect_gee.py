"""
Google Earth Engine data extraction for Karachi.
Extracts: Land Surface Temperature (LST), NDVI, Urban land cover (MODIS).

Requirements:
    pip install earthengine-api
    earthengine authenticate   # run once in terminal
"""

import ee
import pandas as pd
from datetime import datetime
from config import DATA_RAW_DIR

# Karachi geometry (bounding box)
KARACHI_BOUNDS = ee.Geometry.Rectangle([66.85, 24.74, 67.25, 25.10])

DATE_START = "2023-01-01"
DATE_END = datetime.utcnow().strftime("%Y-%m-%d")


def extract_lst(scale: int = 1000) -> pd.DataFrame:
    """MODIS MOD11A1 — daily Land Surface Temperature (LST) at 1km."""
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(KARACHI_BOUNDS)
        .select(["LST_Day_1km", "QC_Day"])
    )

    def image_to_row(img):
        date = img.date().format("YYYY-MM-dd")
        stats = img.select("LST_Day_1km").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=KARACHI_BOUNDS,
            scale=scale,
            maxPixels=1e9,
        )
        lst_raw = stats.get("LST_Day_1km")
        # Convert Kelvin (×0.02 scale factor) to Celsius
        lst_c = ee.Number(lst_raw).multiply(0.02).subtract(273.15)
        return ee.Feature(None, {"date": date, "LST_C": lst_c})

    features = collection.map(image_to_row)
    data = features.getInfo()["features"]

    rows = [
        {
            "date": f["properties"]["date"],
            "LST_C": f["properties"].get("LST_C"),
        }
        for f in data
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_ndvi(scale: int = 500) -> pd.DataFrame:
    """MODIS MOD13A1 — 16-day NDVI at 500m."""
    collection = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(DATE_START, DATE_END)
        .filterBounds(KARACHI_BOUNDS)
        .select("NDVI")
    )

    def image_to_row(img):
        date = img.date().format("YYYY-MM-dd")
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=KARACHI_BOUNDS,
            scale=scale,
            maxPixels=1e9,
        )
        ndvi = ee.Number(stats.get("NDVI")).multiply(0.0001)
        return ee.Feature(None, {"date": date, "NDVI": ndvi})

    features = collection.map(image_to_row)
    data = features.getInfo()["features"]

    rows = [
        {
            "date": f["properties"]["date"],
            "NDVI": f["properties"].get("NDVI"),
        }
        for f in data
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_land_cover(scale: int = 500) -> pd.DataFrame:
    """MODIS MCD12Q1 — annual land cover type at 500m."""
    collection = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate("2020-01-01", DATE_END)
        .filterBounds(KARACHI_BOUNDS)
        .select("LC_Type1")
    )

    def image_to_row(img):
        date = img.date().format("YYYY")
        # Urban = class 13 in IGBP scheme
        urban_mask = img.eq(13)
        total_pixels = img.gt(0).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=KARACHI_BOUNDS,
            scale=scale,
            maxPixels=1e9,
        ).get("LC_Type1")
        urban_pixels = urban_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=KARACHI_BOUNDS,
            scale=scale,
            maxPixels=1e9,
        ).get("LC_Type1")
        urban_fraction = ee.Number(urban_pixels).divide(ee.Number(total_pixels))
        return ee.Feature(None, {"year": date, "urban_fraction": urban_fraction})

    features = collection.map(image_to_row)
    data = features.getInfo()["features"]

    rows = [
        {
            "year": int(f["properties"]["year"]),
            "urban_fraction": f["properties"].get("urban_fraction"),
        }
        for f in data
    ]
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    print("Authenticating with Google Earth Engine...")
    ee.Initialize()

    print("Extracting LST...")
    df_lst = extract_lst()

    print("Extracting NDVI...")
    df_ndvi = extract_ndvi()

    print("Extracting land cover...")
    df_lc = extract_land_cover()

    # Merge LST + NDVI on date (forward-fill NDVI for 16-day gap)
    df = pd.merge(df_lst, df_ndvi, on="date", how="left")
    df = df.sort_values("date")
    df["NDVI"] = df["NDVI"].ffill()

    # Attach urban fraction by year
    df["year"] = df["date"].dt.year
    df = pd.merge(df, df_lc, on="year", how="left")
    df = df.drop(columns=["year"])

    out_path = DATA_RAW_DIR / "gee_karachi.parquet"
    df.to_parquet(out_path, index=False)
    print(f"GEE: {len(df)} rows saved → {out_path}")
    return df


if __name__ == "__main__":
    collect()
