"""
Python GEE: Export daily LST for Karachi to data/raw/karachi_lst.csv
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


def to_celsius(image):
    qc = image.select('QC_Day')
    good = qc.bitwiseAnd(3).eq(0)
    lst = (image.select('LST_Day_1km')
               .multiply(0.02)
               .subtract(273.15)
               .updateMask(good)
               .rename('LST_C'))
    return lst.set('system:time_start', image.get('system:time_start'))


def image_to_feature(image):
    date  = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=KARACHI,
        scale=1000,
        maxPixels=1e9,
    )
    return ee.Feature(None, {'date': date, 'LST_C': stats.get('LST_C')})


def collect() -> pd.DataFrame:
    ee.Initialize(project='project-d87d1ec5-105c-4d29-8f0')
    print("Extracting LST...")

    collection = (
        ee.ImageCollection('MODIS/061/MOD11A1')
        .filterDate(START, END)
        .filterBounds(KARACHI)
        .select(['LST_Day_1km', 'QC_Day'])
        .map(to_celsius)
    )

    features = collection.map(image_to_feature)
    features = features.filter(ee.Filter.notNull(['LST_C']))
    data = features.getInfo()['features']

    rows = [
        {'date': f['properties']['date'], 'LST_C': f['properties']['LST_C']}
        for f in data
    ]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    out = os.path.join(DATA_RAW_DIR, 'karachi_lst.csv')
    df.to_csv(out, index=False)
    print(f"LST: {len(df)} rows → {out}")
    return df


if __name__ == '__main__':
    collect()
