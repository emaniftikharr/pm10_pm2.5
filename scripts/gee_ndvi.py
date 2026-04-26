"""
Python GEE: Export 16-day NDVI composites for Karachi to data/raw/karachi_ndvi.csv
Run once after: earthengine authenticate
"""

import ee
import os
import pandas as pd

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(DATA_RAW_DIR, exist_ok=True)

KARACHI = ee.Geometry.Rectangle([66.85, 24.74, 67.25, 25.10])
START = '2018-01-01'
END   = '2024-12-31'


def scale_ndvi(image):
    qa   = image.select('SummaryQA')
    good = qa.lte(1)
    ndvi = (image.select('NDVI')
                 .multiply(0.0001)
                 .updateMask(good)
                 .rename('NDVI'))
    return ndvi.set('system:time_start', image.get('system:time_start'))


def image_to_feature(image):
    date  = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=KARACHI,
        scale=500,
        maxPixels=1e9,
    )
    return ee.Feature(None, {'date': date, 'NDVI': stats.get('NDVI')})


def collect() -> pd.DataFrame:
    ee.Initialize(project='project-d87d1ec5-105c-4d29-8f0')
    print("Extracting NDVI...")

    collection = (
        ee.ImageCollection('MODIS/061/MOD13A1')
        .filterDate(START, END)
        .filterBounds(KARACHI)
        .select(['NDVI', 'SummaryQA'])
        .map(scale_ndvi)
    )

    features = collection.map(image_to_feature)
    features = features.filter(ee.Filter.notNull(['NDVI']))
    data = features.getInfo()['features']

    rows = [
        {'date': f['properties']['date'], 'NDVI': f['properties']['NDVI']}
        for f in data
    ]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    out = os.path.join(DATA_RAW_DIR, 'karachi_ndvi.csv')
    df.to_csv(out, index=False)
    print(f"NDVI: {len(df)} rows → {out}")
    return df


if __name__ == '__main__':
    collect()
