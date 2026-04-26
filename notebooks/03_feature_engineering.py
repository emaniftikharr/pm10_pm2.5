# Phase 3: Feature Engineering
# Run after: python src/preprocess.py

import sys
sys.path.append('../')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.feature_engineering import run_pipeline

# Run pipeline
df = run_pipeline()

# ── Quick EDA on engineered features ──────────────────────────────────────────

print("\n── Feature Summary ──")
print(df.describe().round(2))

# Plot lag features vs target
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["PM25_lag1", "PM25_lag2", "PM25_lag3"]):
    if col in df.columns and "pm2_5" in df.columns:
        axes[i].scatter(df[col], df["pm2_5"], alpha=0.3, s=5)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("PM2.5")
        axes[i].set_title(f"{col} vs PM2.5")
plt.tight_layout()
plt.savefig("reports/lag_features.png", dpi=150)
plt.show()

# Plot rolling averages
if "AQI_roll7" in df.columns:
    plt.figure(figsize=(14, 4))
    aqi_col = next((c for c in ["AQI", "aqicn_AQI"] if c in df.columns), None)
    if aqi_col:
        plt.plot(df["date"], df[aqi_col], alpha=0.4, label="AQI raw")
    plt.plot(df["date"], df["AQI_roll7"],  label="7-day avg")
    plt.plot(df["date"], df["AQI_roll30"], label="30-day avg")
    plt.legend()
    plt.title("AQI Rolling Averages")
    plt.tight_layout()
    plt.savefig("reports/rolling_averages.png", dpi=150)
    plt.show()

# Distribution of temporal features
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["month", "season", "day_of_week"]):
    df[col].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("reports/temporal_features.png", dpi=150)
plt.show()

print("\nAll plots saved to reports/")
