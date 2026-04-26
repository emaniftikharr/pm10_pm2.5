# Data Sources

All data is for **Karachi, Pakistan** (lat: 24.8607, lon: 67.0011 | bbox: 66.85–67.25°E, 24.74–25.10°N).

---

## 1. Air Quality — Open-Meteo (Primary)

| Field | Detail |
|---|---|
| **Source** | https://air-quality-api.open-meteo.com/v1/air-quality |
| **Date range** | 2025-01-01 → present |
| **Variables** | PM2.5 (µg/m³), PM10 (µg/m³), CO (µg/m³), NO₂ (µg/m³), SO₂ (µg/m³) |
| **Resolution** | Hourly |
| **License** | Open Data (CC BY 4.0) |
| **Local file** | `data/raw/karachi_aqi_raw.csv` |
| **Script** | `src/ingest_data.py` → `fetch_air_quality()` |

---

## 2. Meteorological — Open-Meteo (Primary)

| Field | Detail |
|---|---|
| **Source (archive)** | https://archive-api.open-meteo.com/v1/archive |
| **Source (forecast)** | https://api.open-meteo.com/v1/forecast |
| **Date range** | 2025-01-01 → present |
| **Variables** | Temperature (°C), Dew point (°C), Surface pressure (hPa), Wind speed (m/s), Wind direction (°), Precipitation (mm), Cloud cover (%), Relative humidity (%) |
| **Resolution** | Hourly |
| **License** | Open Data (CC BY 4.0) |
| **Local file** | `data/raw/karachi_aqi_raw.csv` (merged with AQ) |
| **Script** | `src/ingest_data.py` → `fetch_weather()` |

---

## 3. Air Quality — OpenAQ (Supplementary)

| Field | Detail |
|---|---|
| **Source** | https://api.openaq.org/v3/measurements |
| **Date range** | Up to 1 year back (API limit) |
| **Variables** | PM2.5, PM10, NO₂, CO, O₃, SO₂ |
| **Resolution** | Varies by station (hourly typical) |
| **Auth** | API key required → `OPENAQ_API_KEY` in `.env` |
| **License** | Open Data |
| **Local file** | `data/raw/openaq_karachi.parquet` |
| **Script** | `src/collect_openaq.py` |
| **API endpoint** | `GET https://api.openaq.org/v3/measurements?locations_id=<id>&parameters_id=pm25,...` |

---

## 4. AQI — AQICN / WAQI (Supplementary)

| Field | Detail |
|---|---|
| **Source** | https://api.waqi.info/feed/@{uid}/?token=TOKEN |
| **Date range** | Real-time snapshot (no bulk history without paid plan) |
| **Variables** | AQI (0–500 scale), PM2.5, PM10, NO₂, CO, SO₂, O₃ |
| **Resolution** | Hourly snapshots |
| **Auth** | Token required → `AQICN_API_KEY` in `.env` |
| **License** | Non-commercial use only |
| **Local file** | `data/raw/aqicn_karachi.parquet` |
| **Script** | `src/collect_aqicn.py` |
| **Register** | https://aqicn.org/api/ |

---

## 5. Land Surface Temperature (LST) — MODIS via GEE

| Field | Detail |
|---|---|
| **Dataset** | MODIS/061/MOD11A1 |
| **Source** | Google Earth Engine |
| **Date range** | 2018-01-01 → 2024-12-31 |
| **Variables** | LST_Day (°C), quality-filtered |
| **Resolution** | Daily, 1 km spatial |
| **Scale factor** | Raw DN × 0.02 − 273.15 = °C |
| **Quality filter** | QC bits 0–1 == 00 (good quality only) |
| **License** | NASA Open Data |
| **Local file** | `data/raw/karachi_lst.csv` |
| **Scripts** | `scripts/gee_lst.js` (GEE Code Editor), `scripts/gee_lst.py` (Python API) |

---

## 6. NDVI — MODIS via GEE

| Field | Detail |
|---|---|
| **Dataset** | MODIS/061/MOD13A1 |
| **Source** | Google Earth Engine |
| **Date range** | 2018-01-01 → 2024-12-31 |
| **Variables** | NDVI (−1 to 1) |
| **Resolution** | 16-day composite, 500 m spatial |
| **Scale factor** | Raw DN × 0.0001 |
| **Quality filter** | SummaryQA ≤ 1 (good/marginal quality) |
| **License** | NASA Open Data |
| **Local file** | `data/raw/karachi_ndvi.csv` |
| **Scripts** | `scripts/gee_ndvi.js` (GEE Code Editor), `scripts/gee_ndvi.py` (Python API) |

---

## GEE Authentication

```bash
pip install earthengine-api
earthengine authenticate   # opens browser → sign in with Google
```

Register at: https://earthengine.google.com (free for research/non-commercial)

---

## File Summary

| File | Source | Rows (approx) | Format |
|---|---|---|---|
| `karachi_aqi_raw.csv` | Open-Meteo | ~7000 (hourly, 1 yr) | CSV |
| `karachi_lst.csv` | MODIS MOD11A1 | ~2000 (daily, 7 yr) | CSV |
| `karachi_ndvi.csv` | MODIS MOD13A1 | ~160 (16-day, 7 yr) | CSV |
| `openaq_karachi.parquet` | OpenAQ v3 | varies | Parquet |
| `aqicn_karachi.parquet` | AQICN | varies | Parquet |
