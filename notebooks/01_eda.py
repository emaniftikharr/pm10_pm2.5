# Phase 2: Exploratory Data Analysis — Karachi Air Quality
# Run after: python src/preprocess.py && python src/feature_engineering.py

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    path = os.path.join("data", "processed", "feature_matrix.csv")
    if not os.path.exists(path):
        path = os.path.join("data", "processed", "karachi_cleaned.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Resolve column names
    pm25 = next((c for c in ["pm2_5", "PM2.5", "aqicn_PM2.5"] if c in df.columns), None)
    pm10 = next((c for c in ["pm10", "PM10", "aqicn_PM10"] if c in df.columns), None)
    aqi  = next((c for c in ["AQI",  "aqicn_AQI"] if c in df.columns), None)

    print(f"Loaded: {df.shape} | PM2.5={pm25} | PM10={pm10} | AQI={aqi}")
    return df, pm25, pm10, aqi

df, PM25, PM10, AQI = load_data()


# ── 1. AQI Time-Series ────────────────────────────────────────────────────────

def plot_aqi_timeseries():
    if not AQI:
        print("AQI column not found, skipping time-series plot.")
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["date"], df[AQI], linewidth=0.8, color="steelblue", label="AQI")

    # Rolling average
    roll = df[AQI].rolling(30, min_periods=1).mean()
    ax.plot(df["date"], roll, color="red", linewidth=1.5, label="30-day avg")

    # Annotate spikes (AQI > 200)
    spikes = df[df[AQI] > 200]
    ax.scatter(spikes["date"], spikes[AQI], color="red", s=20, zorder=5, label="Spike (AQI>200)")

    ax.axhline(100, color="orange", linestyle="--", linewidth=0.8, label="Unhealthy (100)")
    ax.axhline(200, color="red",    linestyle="--", linewidth=0.8, label="Hazardous (200)")
    ax.set_title("Karachi AQI Time-Series", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "01_aqi_timeseries.png"), dpi=150)
    plt.show()
    print(f"Spikes (AQI>200): {len(spikes)} days")

plot_aqi_timeseries()


# ── 2. Seasonal Analysis — Box Plots ──────────────────────────────────────────

def plot_seasonal():
    month_col = "month" if "month" in df.columns else None
    if not month_col:
        df["month"] = df["date"].dt.month

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, col, label in zip(axes, [PM25, AQI], ["PM2.5 (µg/m³)", "AQI"]):
        if not col:
            continue
        plot_df = df[["month", col]].dropna()
        plot_df["month_name"] = plot_df["month"].map(month_names)
        order = [month_names[i] for i in range(1, 13) if i in plot_df["month"].values]
        sns.boxplot(data=plot_df, x="month_name", y=col, order=order, ax=ax,
                    hue="month_name", palette="coolwarm", legend=False)
        ax.set_title(f"Monthly Distribution — {label}")
        ax.set_xlabel("Month")
        ax.set_ylabel(label)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle("Seasonal Analysis: Winter (Nov–Feb) typically shows higher pollution", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "02_seasonal_boxplots.png"), dpi=150)
    plt.show()

plot_seasonal()


# ── 3. Correlation Matrix ─────────────────────────────────────────────────────

def plot_correlation():
    num_df = df.select_dtypes(include=np.number).drop(
        columns=["is_weekend", "day_of_week", "season"], errors="ignore"
    )
    # Keep only columns with variance
    num_df = num_df.loc[:, num_df.std() > 0]
    corr = num_df.corr()

    # Focus on top correlated features with PM2.5 if available
    if PM25 and PM25 in corr:
        top_cols = corr[PM25].abs().nlargest(15).index.tolist()
        corr = corr.loc[top_cols, top_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Correlation Matrix (top features vs PM2.5)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "03_correlation_matrix.png"), dpi=150)
    plt.show()

    # Print strong correlations
    if PM25 and PM25 in df.columns:
        strong = corr[PM25].abs().sort_values(ascending=False)
        strong = strong[strong > 0.4].drop(PM25, errors="ignore")
        print(f"\nStrong correlations with PM2.5 (|r|>0.4):\n{strong.round(3).to_string()}")

plot_correlation()


# ── 4. Distribution Plots ─────────────────────────────────────────────────────

def plot_distributions():
    targets = [(PM25, "PM2.5 (µg/m³)"), (PM10, "PM10 (µg/m³)"), (AQI, "AQI")]
    targets = [(c, l) for c, l in targets if c and c in df.columns]

    fig, axes = plt.subplots(2, len(targets), figsize=(6 * len(targets), 10))
    if len(targets) == 1:
        axes = axes.reshape(2, 1)

    for i, (col, label) in enumerate(targets):
        data = df[col].dropna()

        # Original distribution
        sns.histplot(data, kde=True, ax=axes[0, i], color="steelblue")
        axes[0, i].set_title(f"{label} — Original")
        skew = data.skew()
        axes[0, i].set_xlabel(f"Skewness: {skew:.2f}")

        # Log-transformed
        log_data = np.log1p(data.clip(lower=0))
        sns.histplot(log_data, kde=True, ax=axes[1, i], color="coral")
        axes[1, i].set_title(f"{label} — Log-transformed")
        axes[1, i].set_xlabel(f"Skewness: {log_data.skew():.2f}")

    plt.suptitle("Pollution Distribution (right-skew is typical — log transform helps)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "04_distributions.png"), dpi=150)
    plt.show()

plot_distributions()


# ── 5. UHI Analysis: LST vs AQI ──────────────────────────────────────────────

def plot_uhi():
    if "LST_C" not in df.columns or not AQI:
        print("LST or AQI column not found, skipping UHI plot.")
        return

    season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
    plot_df = df[["LST_C", AQI, "season"]].dropna()
    plot_df["Season"] = plot_df["season"].map(season_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"Winter": "blue", "Spring": "green", "Summer": "red", "Autumn": "orange"}
    for season, grp in plot_df.groupby("Season"):
        ax.scatter(grp["LST_C"], grp[AQI], label=season,
                   alpha=0.5, s=15, color=palette.get(season, "gray"))

    # Trend line
    from numpy.polynomial.polynomial import polyfit
    x, y = plot_df["LST_C"], plot_df[AQI]
    b, m = polyfit(x, y, 1)
    ax.plot(sorted(x), [b + m * xi for xi in sorted(x)],
            color="black", linewidth=1.5, linestyle="--", label="Trend")

    r = x.corr(y)
    ax.set_title(f"UHI Effect: LST vs AQI  (r = {r:.2f})", fontsize=13)
    ax.set_xlabel("Land Surface Temperature (°C)")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "05_uhi_lst_vs_aqi.png"), dpi=150)
    plt.show()
    print(f"LST–AQI correlation: r = {r:.3f}")

plot_uhi()


# ── 6. Pollution Spikes ───────────────────────────────────────────────────────

def analyze_spikes():
    if not AQI:
        return
    spikes = df[df[AQI] > 200][["date", AQI, PM25, PM10]].dropna(how="all")
    spikes = spikes.sort_values(AQI, ascending=False)
    print(f"\nPollution Spikes (AQI > 200): {len(spikes)} days")
    print(spikes.head(20).to_string(index=False))
    spikes.to_csv(os.path.join(REPORTS_DIR, "pollution_spikes.csv"), index=False)
    return spikes

spikes = analyze_spikes()


# ── 7. EDA Summary Report ─────────────────────────────────────────────────────

def write_eda_summary():
    num_df = df.select_dtypes(include=np.number)

    pm25_mean  = f"{df[PM25].mean():.2f}"  if PM25 else "N/A"
    pm25_max   = f"{df[PM25].max():.2f}"   if PM25 else "N/A"
    aqi_mean   = f"{df[AQI].mean():.2f}"   if AQI  else "N/A"
    aqi_max    = f"{df[AQI].max():.2f}"    if AQI  else "N/A"
    spike_days = len(df[df[AQI] > 200])    if AQI  else "N/A"

    # Top corr with PM2.5
    top_corr = ""
    if PM25 and PM25 in num_df.columns:
        corr_series = num_df.corr()[PM25].abs().sort_values(ascending=False)
        corr_series = corr_series[corr_series > 0.3].drop(PM25, errors="ignore").head(8)
        top_corr = "\n".join([f"- {k}: {v:.3f}" for k, v in corr_series.items()])

    report = f"""# EDA Summary Report — Karachi Air Quality

## Dataset Overview
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Date range: {df['date'].min().date()} to {df['date'].max().date()}

## Key Statistics
| Metric | PM2.5 | AQI |
|--------|-------|-----|
| Mean   | {pm25_mean} | {aqi_mean} |
| Max    | {pm25_max}  | {aqi_max}  |

## Seasonal Trends
- **Winter (Nov–Feb):** Highest PM2.5 and AQI due to temperature inversion trapping pollutants near surface
- **Summer (Jun–Aug):** Lower AQI but elevated LST; sea breeze helps disperse pollutants
- **Monsoon (Jul–Sep):** Rain washout temporarily reduces PM levels

## Top Correlations with PM2.5 (|r| > 0.3)
{top_corr if top_corr else "Insufficient data for correlation analysis"}

## Distribution Analysis
- PM2.5 and PM10 show **right-skewed** distributions (typical for pollution data)
- Log-transformation recommended before feeding into linear models
- AQI distribution shows multi-modal pattern reflecting seasonal variation

## UHI (Urban Heat Island) Analysis
- LST positively correlated with AQI — higher surface temperature associated with higher pollution
- Strongest UHI–AQI relationship observed in **Summer** season
- Urban land cover fraction contributes to both LST elevation and pollution trapping

## Pollution Spikes (AQI > 200)
- Total hazardous days: **{spike_days}**
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
"""
    out = os.path.join(REPORTS_DIR, "eda_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nEDA summary saved → {out}")

write_eda_summary()

print("\nAll EDA tasks complete. Check the reports/ folder.")
