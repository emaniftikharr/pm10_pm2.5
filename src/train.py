"""
Phase 4: Model Training, Tuning & Evaluation
- Linear Regression, Random Forest, XGBoost, SVR
- TimeSeriesSplit cross-validation
- GridSearchCV hyperparameter tuning
- 3-day recursive forecast
- Multi-output prediction (AQI, PM2.5, PM10)
- Ablation study (with vs without satellite features)
- Feature importance plots
- Save all models + tuning logs
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

MODELS_DIR    = "models"
REPORTS_DIR   = "reports"
PROCESSED_DIR = os.path.join("data", "processed")
for d in [MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

TARGETS = ["pm2_5", "pm10", "aqicn_AQI"]
SATELLITE_COLS = ["LST_C", "NDVI", "LST_roll30"]
EXCLUDE_COLS   = ["date"] + TARGETS + [
    c for c in ["AQI", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
                "aqicn_NO2", "aqicn_CO", "aqicn_SO2", "aqicn_O3",
                "aqicn_PM2.5", "aqicn_PM10"]
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_splits():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), parse_dates=["date"])
    val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"),   parse_dates=["date"])
    test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"),  parse_dates=["date"])
    return train, val, test


def get_features(df, targets, exclude):
    exclude_present = [c for c in exclude if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude_present]
    return feature_cols


def metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label:30s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def resolve_targets(df):
    found = []
    for t in TARGETS:
        if t in df.columns:
            found.append(t)
    if not found:
        raise ValueError(f"None of {TARGETS} found in dataframe.")
    return found


# ── 1. BASELINE: Linear Regression ───────────────────────────────────────────

def train_linear(X_train, y_train, X_val, y_val, target):
    print(f"\n[LR] {target}")
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred  = model.predict(X_val)
    result = metrics(y_val, pred, "LinearRegression")
    result["model"] = "LinearRegression"
    result["target"] = target
    joblib.dump(model, os.path.join(MODELS_DIR, f"lr_{target}.pkl"))
    return model, result


# ── 2. Random Forest ──────────────────────────────────────────────────────────

def train_rf(X_train, y_train, X_val, y_val, target):
    print(f"\n[RF] {target}")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred  = model.predict(X_val)
    result = metrics(y_val, pred, "RandomForest(default)")
    result["model"] = "RandomForest"
    result["target"] = target
    return model, result


# ── 3. XGBoost ────────────────────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_val, y_val, target):
    print(f"\n[XGB] {target}")
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    pred   = model.predict(X_val)
    result = metrics(y_val, pred, "XGBoost(default)")
    result["model"] = "XGBoost"
    result["target"] = target
    return model, result


# ── 4. SVR ────────────────────────────────────────────────────────────────────

def train_svr(X_train, y_train, X_val, y_val, target):
    print(f"\n[SVR] {target}")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=100, gamma=0.1)),
    ])
    pipe.fit(X_train, y_train)
    pred   = pipe.predict(X_val)
    result = metrics(y_val, pred, "SVR(rbf)")
    result["model"] = "SVR"
    result["target"] = target
    joblib.dump(pipe, os.path.join(MODELS_DIR, f"svr_{target}.pkl"))
    return pipe, result


# ── 5. Hyperparameter Tuning ──────────────────────────────────────────────────

def tune_models(X_train, y_train, target):
    print(f"\n[TUNE] {target}")
    tscv = TimeSeriesSplit(n_splits=5)
    logs = []

    # Tune Random Forest
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [5, 10, None],
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"  RF best params : {rf_search.best_params_}  score={-rf_search.best_score_:.3f}")
    joblib.dump(best_rf, os.path.join(MODELS_DIR, f"rf_best_{target}.pkl"))

    for p, s in zip(rf_search.cv_results_["params"],
                    -rf_search.cv_results_["mean_test_score"]):
        logs.append({"model": "RF", "target": target, **p, "cv_mae": s})

    # Tune XGBoost
    xgb_grid = {
        "learning_rate": [0.01, 0.1],
        "max_depth":     [3, 6, 9],
        "n_estimators":  [100, 200],
    }
    xgb_search = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
        xgb_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"  XGB best params: {xgb_search.best_params_}  score={-xgb_search.best_score_:.3f}")
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, f"xgb_best_{target}.pkl"))

    for p, s in zip(xgb_search.cv_results_["params"],
                    -xgb_search.cv_results_["mean_test_score"]):
        logs.append({"model": "XGB", "target": target, **p, "cv_mae": s})

    return best_rf, best_xgb, pd.DataFrame(logs)


# ── 6. Recursive 3-Day Forecast ───────────────────────────────────────────────

def recursive_forecast(model, X_val, y_val, feature_cols, target, horizon=3):
    print(f"\n[FORECAST] 3-day recursive forecast for {target}")
    results = []
    lag_cols = [c for c in feature_cols if "lag1" in c or "lag_1" in c]

    for h in range(1, horizon + 1):
        preds, actuals = [], []
        for i in range(len(X_val) - horizon):
            row = X_val.iloc[[i]].copy()
            # For h>1: inject previous prediction as lag1
            if h > 1 and lag_cols:
                row[lag_cols[0]] = preds[-1] if preds else row[lag_cols[0]].values[0]
            pred = model.predict(row)[0]
            actual = y_val.iloc[i + h - 1] if i + h - 1 < len(y_val) else np.nan
            preds.append(pred)
            actuals.append(actual)

        valid = [(a, p) for a, p in zip(actuals, preds) if not np.isnan(a)]
        if valid:
            a_arr = np.array([v[0] for v in valid])
            p_arr = np.array([v[1] for v in valid])
            mae  = mean_absolute_error(a_arr, p_arr)
            rmse = np.sqrt(mean_squared_error(a_arr, p_arr))
            r2   = r2_score(a_arr, p_arr)
            print(f"  Day {h}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
            results.append({"horizon": h, "MAE": mae, "RMSE": rmse, "R2": r2})

    return pd.DataFrame(results)


# ── 7. Multi-Output Prediction ────────────────────────────────────────────────

def train_multioutput(X_train, y_train_multi, X_val, y_val_multi, target_cols):
    print(f"\n[MULTI-OUTPUT] Targets: {target_cols}")

    # Approach 1: MultiOutputRegressor
    mo_model = MultiOutputRegressor(
        XGBRegressor(objective="reg:squarederror", n_estimators=200,
                     random_state=42, verbosity=0),
        n_jobs=-1
    )
    mo_model.fit(X_train, y_train_multi)
    pred_mo = mo_model.predict(X_val)
    print("  MultiOutputRegressor(XGB):")
    for i, col in enumerate(target_cols):
        metrics(y_val_multi.iloc[:, i], pred_mo[:, i], f"  {col}")

    joblib.dump(mo_model, os.path.join(MODELS_DIR, "xgb_multioutput.pkl"))
    return mo_model


# ── 8. Ablation Study ─────────────────────────────────────────────────────────

def ablation_study(X_train, y_train, X_val, y_val, target):
    print(f"\n[ABLATION] {target} — with vs without satellite features")
    sat_present = [c for c in SATELLITE_COLS if c in X_train.columns]

    results = []
    for label, drop_sat in [("With satellite", False), ("Without satellite", True)]:
        Xtr = X_train.drop(columns=sat_present) if drop_sat and sat_present else X_train
        Xvl = X_val.drop(columns=sat_present)   if drop_sat and sat_present else X_val
        model = XGBRegressor(objective="reg:squarederror", n_estimators=200,
                             random_state=42, verbosity=0)
        model.fit(Xtr, y_train)
        pred = model.predict(Xvl)
        r = metrics(y_val, pred, label)
        r["label"] = label
        results.append(r)

    diff = results[0]["MAE"] - results[1]["MAE"]
    print(f"  Satellite feature impact on MAE: {diff:+.3f} (negative = satellite helps)")
    return pd.DataFrame(results)


# ── 9. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols, target, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance — {target} (top {top_n})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"importance_{target}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("Phase 4: Model Training & Evaluation")
    print("=" * 60)

    train, val, test = load_splits()
    targets = resolve_targets(train)
    feature_cols = get_features(train, targets, EXCLUDE_COLS)
    feature_cols = [c for c in feature_cols if c in train.columns]

    print(f"\nTargets   : {targets}")
    print(f"Features  : {len(feature_cols)} columns")
    print(f"Train rows: {len(train)} | Val rows: {len(val)} | Test rows: {len(test)}")

    all_results  = []
    tune_logs    = []
    forecast_dfs = []

    for target in targets:
        if target not in train.columns:
            continue

        X_train = train[feature_cols].fillna(0)
        y_train = train[target]
        X_val   = val[feature_cols].fillna(0)
        y_val   = val[target]

        # Skip if insufficient data
        print(f"  X_train={len(X_train)} rows | X_val={len(X_val)} rows")
        if len(X_val) < 5:
            print(f"\nSkipping {target} — not enough validation data ({len(X_val)} rows)")
            continue

        print(f"\n{'='*50}\nTarget: {target}\n{'='*50}")

        # 1. Baseline models
        _, r = train_linear(X_train, y_train, X_val, y_val, target)
        all_results.append(r)

        rf_model, r = train_rf(X_train, y_train, X_val, y_val, target)
        all_results.append(r)

        xgb_model, r = train_xgb(X_train, y_train, X_val, y_val, target)
        all_results.append(r)

        _, r = train_svr(X_train, y_train, X_val, y_val, target)
        all_results.append(r)

        # 2. Tuning
        best_rf, best_xgb, log_df = tune_models(X_train, y_train, target)
        tune_logs.append(log_df)

        pred_rf  = best_rf.predict(X_val)
        pred_xgb = best_xgb.predict(X_val)
        r_rf  = metrics(y_val, pred_rf,  f"RF(tuned)")
        r_xgb = metrics(y_val, pred_xgb, f"XGB(tuned)")
        r_rf["model"]  = "RF_tuned";  r_rf["target"]  = target
        r_xgb["model"] = "XGB_tuned"; r_xgb["target"] = target
        all_results += [r_rf, r_xgb]

        # 3. 3-day recursive forecast (use best XGB)
        fc_df = recursive_forecast(best_xgb, X_val, y_val, feature_cols, target)
        fc_df["target"] = target
        forecast_dfs.append(fc_df)

        # 4. Ablation
        abl_df = ablation_study(X_train, y_train, X_val, y_val, target)

        # 5. Feature importance
        plot_feature_importance(best_rf,  feature_cols, f"{target}_rf",  top_n=20)
        plot_feature_importance(best_xgb, feature_cols, f"{target}_xgb", top_n=20)

    # Multi-output (all targets together)
    multi_targets = [t for t in targets if t in train.columns and t in val.columns]
    if len(multi_targets) > 1:
        X_train_mo = train[feature_cols].fillna(0)
        X_val_mo   = val[feature_cols].fillna(0)
        y_train_mo = train[multi_targets]
        y_val_mo   = val[multi_targets]
        if len(X_val_mo) >= 5:
            train_multioutput(X_train_mo, y_train_mo, X_val_mo, y_val_mo, multi_targets)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_results.csv"), index=False)
    print(f"\nModel results saved → reports/model_results.csv")

    if tune_logs:
        pd.concat(tune_logs).to_csv(
            os.path.join(REPORTS_DIR, "tuning_logs.csv"), index=False)
        print("Tuning logs saved  → reports/tuning_logs.csv")

    if forecast_dfs:
        pd.concat(forecast_dfs).to_csv(
            os.path.join(REPORTS_DIR, "forecast_results.csv"), index=False)
        print("Forecast results   → reports/forecast_results.csv")

    # Print final leaderboard
    print("\n" + "=" * 60)
    print("LEADERBOARD (Validation Set)")
    print("=" * 60)
    if results_df.empty:
        print("No results — check train/val split sizes above.")
    else:
        sort_cols = [c for c in ["target", "MAE"] if c in results_df.columns]
        print(results_df.sort_values(sort_cols).to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_pipeline()
