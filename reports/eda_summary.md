# EDA Summary Report — Karachi Air Quality

## Dataset Overview
- Shape: 479 rows × 49 columns
- Date range: 2025-01-03 to 2026-04-29

## Key Statistics
| Metric | PM2.5 | AQI |
|--------|-------|-----|
| Mean   | 29.13 | 142.77 |
| Max    | 69.30  | 164.00  |

## Seasonal Trends
- **Winter (Nov–Feb):** Highest PM2.5 and AQI due to temperature inversion trapping pollutants near surface
- **Summer (Jun–Aug):** Lower AQI but elevated LST; sea breeze helps disperse pollutants
- **Monsoon (Jul–Sep):** Rain washout temporarily reduces PM levels

## Top Correlations with PM2.5 (|r| > 0.3)
- PM25_lag1: 0.738
- PM25_roll7: 0.711
- sulphur_dioxide: 0.684
- carbon_monoxide: 0.641
- nitrogen_dioxide: 0.612
- PM25_lag2: 0.545
- wind_speed_10m: 0.496
- pm10: 0.460

## Distribution Analysis
- PM2.5 and PM10 show **right-skewed** distributions (typical for pollution data)
- Log-transformation recommended before feeding into linear models
- AQI distribution shows multi-modal pattern reflecting seasonal variation

## UHI (Urban Heat Island) Analysis
- LST positively correlated with AQI — higher surface temperature associated with higher pollution
- Strongest UHI–AQI relationship observed in **Summer** season
- Urban land cover fraction contributes to both LST elevation and pollution trapping

## Pollution Spikes (AQI > 200)
- Total hazardous days: **0**
- Likely causes: industrial emissions, vehicular congestion, dust storms (Karachi coastal dust)
- See: reports/pollution_spikes.csv for full list

## Figures Generated
| File | Description |
|------|-------------|
| 01_aqi_timeseries.png | Full AQI time-series with spikes annotated |
| 02_seasonal_boxplots.png | Monthly PM2.5 and AQI distributions |
| 03_correlation_matrix.png | Feature correlation heatmap |
| 04_distributions.png | Histogram + KDE, original vs log-transformed |
| 05_uhi_lst_vs_aqi.png | LST vs AQI scatter colored by season |
| pollution_spikes.csv | All days with AQI > 200 |

## Recommended Next Steps
1. Apply log-transform to PM2.5, PM10 for linear models
2. Use lag features (lag1, lag2, lag3) — strong temporal autocorrelation expected
3. Season and month are important categorical features
4. LST is a useful predictor — retain in feature matrix
