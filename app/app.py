import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR, TARGET_VARIABLES

st.set_page_config(page_title="Air Quality Predictor", layout="wide")
st.title("Air Quality Index Predictor")
st.markdown("Predict **AQI**, **PM2.5**, and **PM10** levels based on environmental inputs.")

st.sidebar.header("Input Parameters")

# Placeholder inputs — update feature names after EDA
temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 15)
no2 = st.sidebar.number_input("NO2 (µg/m³)", 0.0, 500.0, 40.0)
so2 = st.sidebar.number_input("SO2 (µg/m³)", 0.0, 500.0, 20.0)
co = st.sidebar.number_input("CO (mg/m³)", 0.0, 50.0, 1.0)

if st.sidebar.button("Predict"):
    st.info("Model not yet trained. Complete Phase 3 & 4 first.")
