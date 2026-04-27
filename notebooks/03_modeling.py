# Phase 4: Model Training, Tuning & Evaluation

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib

from src.train import run_pipeline, load_splits, get_features, EXCLUDE_COLS, MODELS_DIR, REPORTS_DIR

# ── Run full training pipeline ────────────────────────────────────────────────
results_df = run_pipeline()

# ── Plot: Model Comparison per Target ────────────────────────────────────────
def plot_model_comparison():
    df = pd.read_csv(os.path.join(REPORTS_DIR, "model_results.csv"))
    targets = df["target"].unique()

    fig, axes = plt.subplots(len(targets), 3, figsize=(16, 5 * len(targets)))
    if len(targets) == 1:
        axes = axes.reshape(1, 3)

    for i, target in enumerate(targets):
        sub = df[df["target"] == target].sort_values("MAE")
        for j, metric in enumerate(["MAE", "RMSE", "R2"]):
            ax = axes[i, j]
            colors = ["steelblue" if v != sub[metric].min() else "coral"
                      for v in sub[metric]] if metric != "R2" else \
                     ["steelblue" if v != sub[metric].max() else "coral"
                      for v in sub[metric]]
            ax.barh(sub["model"], sub[metric], color=colors)
            ax.set_title(f"{target} — {metric}")
            ax.set_xlabel(metric)

    plt.suptitle("Model Comparison (highlighted = best)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved → reports/model_comparison.png")

plot_model_comparison()


# ── Plot: 3-Day Forecast Horizon ──────────────────────────────────────────────
def plot_forecast():
    path = os.path.join(REPORTS_DIR, "forecast_results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    targets = df["target"].unique()

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = df[df["target"] == target]
        ax.plot(sub["horizon"], sub["MAE"],  marker="o", label="MAE")
        ax.plot(sub["horizon"], sub["RMSE"], marker="s", label="RMSE")
        ax.set_title(f"3-Day Forecast Degradation — {target}")
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_xticks([1, 2, 3])
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "forecast_horizon.png"), dpi=150)
    plt.close()
    print("Saved → reports/forecast_horizon.png")

plot_forecast()


# ── Plot: Actual vs Predicted (best model) ────────────────────────────────────
def plot_actual_vs_predicted():
    train, val, test = load_splits()
    feature_cols = get_features(train, [], EXCLUDE_COLS)
    feature_cols = [c for c in feature_cols if c in train.columns]

    targets = [t for t in ["pm2_5", "pm10", "aqicn_AQI"] if t in val.columns]
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        model_path = os.path.join(MODELS_DIR, f"xgb_best_{target}.pkl")
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        X_val = val[feature_cols].fillna(0)
        y_val = val[target]
        pred  = model.predict(X_val)

        ax.scatter(y_val, pred, alpha=0.4, s=15, color="steelblue")
        lims = [min(y_val.min(), pred.min()), max(y_val.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {target}")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("Saved → reports/actual_vs_predicted.png")

plot_actual_vs_predicted()

print("\nAll modeling tasks complete.")
print("Check reports/ for figures and CSVs.")
print("Check models/ for saved .pkl files.")
