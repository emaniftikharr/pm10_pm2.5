# Air Quality Prediction — Project Proposal

## Problem Statement
Air pollution is a major public health concern. This project builds a machine learning system to predict air quality levels (AQI, PM2.5, PM10) using environmental and meteorological data, enabling early warnings and informed decisions.

## Target Variables
| Variable | Description | Unit |
|----------|-------------|------|
| AQI | Air Quality Index | 0–500 scale |
| PM2.5 | Fine particulate matter (≤2.5µm) | µg/m³ |
| PM10 | Coarse particulate matter (≤10µm) | µg/m³ |

## Scope
- Historical air quality + weather data (public datasets: OpenAQ, IQAir, CPCB)
- Feature engineering from pollutant readings and meteorological variables
- Regression models: Linear Regression, Random Forest, XGBoost
- Evaluation: MAE, RMSE, R²
- Deployment: Streamlit web app for real-time prediction

## Project Phases
| Phase | Task | Week |
|-------|------|------|
| 1 | Setup & Planning | 1 |
| 2 | Data Collection & EDA | 2 |
| 3 | Preprocessing & Feature Engineering | 3 |
| 4 | Model Training & Evaluation | 4 |
| 5 | Streamlit App + Final Report | 5 |

## Tech Stack
- Python 3.10+, pandas, numpy, scikit-learn, XGBoost
- Visualization: matplotlib, seaborn, plotly
- Deployment: Streamlit
