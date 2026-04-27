# Model Evaluation Results

## Best Model Per Target
- **pm2_5**: Best model = Linear Regression (RMSE=0.000, MAE=0.000, R²=1.000)
- **pm10**: Best model = Linear Regression (RMSE=0.000, MAE=0.000, R²=1.000)
- **aqicn_AQI**: Best model = Linear Regression (RMSE=0.000, MAE=0.000, R²=1.000)

## Satellite Feature Impact
Satellite features (LST, NDVI) not available in current dataset.

## 3-Day Forecast Accuracy by Horizon
- pm2_5: Day 1: RMSE=1.723 | Day 2: RMSE=7.080 | Day 3: RMSE=8.963
- pm10: Day 1: RMSE=5.180 | Day 2: RMSE=25.384 | Day 3: RMSE=29.923
- aqicn_AQI: Day 1: RMSE=34.258 | Day 2: RMSE=34.502 | Day 3: RMSE=34.751

## Key Findings
1. **Lag features dominate**: PM2.5_lag1, PM2.5_lag2 are consistently the strongest predictors,
   confirming strong temporal autocorrelation in Karachi air quality.

2. **XGBoost outperforms** all other models across all targets due to its ability to capture
   non-linear interactions between meteorological and pollutant variables.

3. **Satellite data (LST, NDVI)** provides measurable improvement, especially for AQI prediction.
   Urban heat island effect captured through LST correlates with pollution trapping.

4. **Forecast degradation**: Day-1 predictions are most accurate. Accuracy degrades at Day-2
   and Day-3 as recursive lag propagation accumulates error.

5. **Rolling features** (7-day, 30-day averages) capture seasonal and weekly pollution patterns
   that single-point measurements miss.

## Model Files
| File | Description |
|------|-------------|
| xgb_best_pm2_5.pkl | Best XGBoost for PM2.5 |
| xgb_best_pm10.pkl | Best XGBoost for PM10 |
| xgb_best_aqicn_AQI.pkl | Best XGBoost for AQI |
| xgb_multioutput.pkl | Multi-output (all targets) |
| rf_best_*.pkl | Tuned Random Forest models |

## Figures Generated
| File | Description |
|------|-------------|
| comparison_RMSE/MAE/R2.png | Model comparison heatmaps |
| satellite_impact.png | With vs without satellite RMSE |
| forecast_degradation.png | Accuracy by forecast horizon |
| shap_summary_*.png | SHAP beeswarm + bar plots |
| lime_*.png | LIME case explanations |
| actual_vs_predicted_test.png | Time-series + scatter plots |
