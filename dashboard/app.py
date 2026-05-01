"""
AQI Dashboard — Karachi Climate Intelligence Center
Dark aerospace aesthetic: animated gauge, glassmorphism, Bloomberg-style charts.
"""
import math
import os
import smtplib
import warnings
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(ROOT, "models")
FEATURE_STORE = os.path.join(ROOT, "data", "processed", "feature_store.csv")

# ── Location ───────────────────────────────────────────────────────────────────
LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"
AQ_URL       = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

MET_COLS = [
    "temperature_2m", "dew_point_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "precipitation",
    "cloud_cover", "relative_humidity_2m",
]
POLLUTANT_COLS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]

_TARGETS     = ["pm2_5", "pm10"]
_EXCLUDE_SET = set(["date"] + _TARGETS + [
    "AQI", "PM2.5", "PM10", "aqicn_AQI", "aqicn_PM2.5", "aqicn_PM10",
    "NO2", "CO", "SO2", "O3", "aqicn_NO2", "aqicn_CO", "aqicn_SO2",
])

# ── AQI ────────────────────────────────────────────────────────────────────────
_PM25_BP = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5,250.4, 201,  300),
    (250.5,350.4, 301,  400),
    (350.5,500.4, 401,  500),
]

# Design tokens — semantic AQI palette
AQI_CATS = [
    (0,   50,  "Good",                          "#00ff88", "#000"),
    (51,  100, "Moderate",                      "#ffb347", "#000"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff6b35", "#fff"),
    (151, 200, "Unhealthy",                     "#ff2244", "#fff"),
    (201, 300, "Very Unhealthy",                "#9b59b6", "#fff"),
    (301, 500, "Hazardous",                     "#7e0023", "#fff"),
]

FEATURE_ICONS = {
    "temperature_2m": "🌡️", "dew_point_2m": "💧", "surface_pressure": "🔵",
    "wind_speed_10m": "💨", "wind_direction_10m": "🧭", "precipitation": "🌧️",
    "cloud_cover": "☁️", "relative_humidity_2m": "💦", "carbon_monoxide": "🏭",
    "nitrogen_dioxide": "⚗️", "sulphur_dioxide": "🌋",
    "pm2_5_lag1": "📅", "pm2_5_lag2": "📅", "pm2_5_lag3": "📅",
    "pm10_lag1": "📆", "pm10_lag2": "📆", "pm10_lag3": "📆",
    "PM25_roll7": "📊", "PM25_roll30": "📈", "PM10_roll7": "📉",
    "month": "🗓️", "season": "🍂", "day_of_week": "📋",
    "day_of_year": "🔢", "is_weekend": "🏖️",
    "pm2_5_change": "📐", "pm10_change": "📏",
}

# Plotly base layout — Bloomberg terminal dark
_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, monospace", color="rgba(255,255,255,0.65)", size=11),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="Space Grotesk, monospace", size=10),
        title_font=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="Space Grotesk, monospace", size=10),
        title_font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="rgba(5,8,16,0.97)",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(family="Space Grotesk, monospace", size=12, color="white"),
    ),
    margin=dict(t=40, b=10, l=10, r=10),
)


def pm25_to_aqi(pm25: float) -> float:
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, a_lo, a_hi in _PM25_BP:
        if c_lo <= pm25 <= c_hi:
            return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
    return 500.0


def aqi_category(aqi: float) -> tuple[str, str, str]:
    for lo, hi, lbl, bg, tc in AQI_CATS:
        if lo <= aqi <= hi:
            return lbl, bg, tc
    return "Hazardous", "#7e0023", "#fff"


# ── SVG gauge geometry ─────────────────────────────────────────────────────────
_CX, _CY, _RO, _RI = 160, 155, 128, 86

def _arc(cx, cy, ro, ri, a_start, a_end):
    """SVG donut arc path — angles in math convention (0=right, CCW positive)."""
    s, e = math.radians(a_start), math.radians(a_end)
    pts = [(cx + r * math.cos(t), cy - r * math.sin(t))
           for r in (ro, ri) for t in (s, e)]
    ox1, oy1 = pts[0]; ox2, oy2 = pts[1]
    ix1, iy1 = pts[2]; ix2, iy2 = pts[3]
    la = 1 if abs(a_start - a_end) > 180 else 0
    return (f"M {ox1:.2f},{oy1:.2f} "
            f"A {ro},{ro} 0 {la},1 {ox2:.2f},{oy2:.2f} "
            f"L {ix2:.2f},{iy2:.2f} "
            f"A {ri},{ri} 0 {la},0 {ix1:.2f},{iy1:.2f} Z")


_GAUGE_SEG = [
    (180, 162, "#00ff88"),  # Good        0-50
    (162, 144, "#ffb347"),  # Moderate   50-100
    (144, 126, "#ff6b35"),  # USG       100-150
    (126, 108, "#ff2244"),  # Unhealthy 150-200
    (108,  72, "#9b59b6"),  # Very      200-300
    ( 72,   0, "#7e0023"),  # Hazardous 300-500
]
_SEG_PATHS = [(_arc(_CX, _CY, _RO, _RI, s, e), c) for s, e, c in _GAUGE_SEG]


# ── Resource loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pm25   = joblib.load(os.path.join(MODELS_DIR, "xgb_best_pm2_5.pkl"))
    pm10   = joblib.load(os.path.join(MODELS_DIR, "xgb_best_pm10.pkl"))
    sc     = os.path.join(MODELS_DIR, "scaler_standard.pkl")
    return pm25, pm10, (joblib.load(sc) if os.path.exists(sc) else None)


@st.cache_data(ttl=3600)
def load_feature_store() -> pd.DataFrame:
    return (pd.read_csv(FEATURE_STORE, parse_dates=["date"])
              .sort_values("date").reset_index(drop=True))


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _EXCLUDE_SET]


# ── API ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def _wx(start: str, end: str) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=16)).strftime("%Y-%m-%d")
    url = ARCHIVE_URL if start < cutoff else FORECAST_URL
    r = requests.get(url, params={
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(MET_COLS),
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    }, timeout=30)
    r.raise_for_status()
    return r.json().get("hourly", {})


@st.cache_data(ttl=1800)
def _aq(start: str, end: str) -> dict:
    r = requests.get(AQ_URL, params={
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(POLLUTANT_COLS),
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    }, timeout=30)
    r.raise_for_status()
    return r.json().get("hourly", {})


# ── Feature engineering ────────────────────────────────────────────────────────
def build_row(dt: datetime, fs: pd.DataFrame, scaler) -> pd.Series | None:
    date_s = dt.strftime("%Y-%m-%d")
    start_s = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        wx_df = pd.DataFrame(_wx(start_s, date_s))
        aq_df = pd.DataFrame(_aq(start_s, date_s))
    except Exception:
        return None
    if wx_df.empty or aq_df.empty:
        return None
    wx_df["time"] = pd.to_datetime(wx_df["time"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])
    merged = pd.merge(aq_df, wx_df, on="time", how="inner")
    if merged.empty:
        return None
    daily = (merged.set_index("time").resample("1D").mean(numeric_only=True)
             .reset_index().rename(columns={"time": "date"}))
    match = daily[daily["date"].dt.date == dt.date()]
    row = (match.iloc[0] if not match.empty else daily.iloc[-1]).copy()

    hist = fs[fs["date"] < pd.Timestamp(dt.date())].sort_values("date")
    p25h = hist["pm2_5"].values if "pm2_5" in hist.columns else np.array([])
    p10h = hist["pm10"].values  if "pm10"  in hist.columns else np.array([])

    for lag in range(1, 4):
        row[f"pm2_5_lag{lag}"] = float(p25h[-lag]) if len(p25h) >= lag else 0.0
        row[f"pm10_lag{lag}"]  = float(p10h[-lag]) if len(p10h) >= lag else 0.0
    row["PM25_roll7"]  = float(p25h[-7:].mean())  if len(p25h) >= 1 else 0.0
    row["PM25_roll30"] = float(p25h[-30:].mean()) if len(p25h) >= 1 else 0.0
    row["PM10_roll7"]  = float(p10h[-7:].mean())  if len(p10h) >= 1 else 0.0

    ts = pd.Timestamp(dt.date()); m = ts.month
    row["month"]       = m
    row["day_of_week"] = ts.dayofweek
    row["is_weekend"]  = int(ts.dayofweek >= 5)
    row["day_of_year"] = ts.dayofyear
    row["season"]      = 1 if m in [12,1,2] else (2 if m in [3,4,5] else (3 if m in [6,7,8] else 4))

    prev25 = float(p25h[-1]) if len(p25h) > 0 else 0.0
    prev10 = float(p10h[-1]) if len(p10h) > 0 else 0.0
    row["pm2_5_change"] = float(row.get("pm2_5", prev25)) - prev25
    row["pm10_change"]  = float(row.get("pm10",  prev10)) - prev10

    if scaler is not None:
        met = [c for c in MET_COLS if c in row.index]
        if met:
            try:
                v = scaler.transform(pd.DataFrame(
                    [row[met].astype(float).values], columns=met))
                for i, c in enumerate(met):
                    row[c] = v[0][i]
            except Exception:
                pass
    return row


def predict(row: pd.Series, fcols: list, m25, m10) -> tuple[float, float]:
    X = pd.DataFrame([row.reindex(fcols).fillna(0).values], columns=fcols)
    return max(0.0, float(m25.predict(X)[0])), max(0.0, float(m10.predict(X)[0]))


# ── Email ──────────────────────────────────────────────────────────────────────
def send_alert(aqi: float, pm25: float) -> bool:
    sender = os.getenv("ALERT_EMAIL"); pwd = os.getenv("ALERT_PASSWORD")
    to     = os.getenv("ALERT_TO", sender)
    if not sender or not pwd:
        return False
    try:
        msg = MIMEText(
            f"HAZARDOUS AIR QUALITY — Karachi\n\n"
            f"AQI: {aqi:.0f}  PM2.5: {pm25:.1f} µg/m³\n"
            f"Time: {datetime.now(pytz.timezone(TIMEZONE)).strftime('%Y-%m-%d %H:%M %Z')}\n\n"
            f"Avoid all outdoor activity. Wear N95 masks. Keep windows closed.", "plain")
        msg["Subject"] = f"Karachi AQI Alert: {aqi:.0f} — Hazardous"
        msg["From"] = sender; msg["To"] = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(sender, pwd); s.sendmail(sender, to, msg.as_string())
        return True
    except Exception as e:
        st.warning(f"Email failed: {e}"); return False


# ── CSS injection ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;700&family=Syne:wght@700;800&display=swap');

/* ── Background ──────────────────────────────────────────────────────────── */
.stApp, .main .block-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #050810 100%) !important;
}
section[data-testid="stSidebar"] {
    background: rgba(8,12,22,0.98) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(20px) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 2rem; }

/* ── Typography ──────────────────────────────────────────────────────────── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
h1 { font-size: 1.9rem !important; font-weight: 800 !important; }
h2 { font-size: 1.3rem !important; font-weight: 700 !important; }
h3 { font-size: 1.05rem !important; font-weight: 700 !important; }
p, div, li, span { font-family: 'Inter', sans-serif; }

/* ── Metric cards — glassmorphism ────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    min-height: 90px !important;
}
[data-testid="stMetric"]:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.14) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    opacity: 0.55 !important;
    margin-bottom: 4px !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Plotly charts ───────────────────────────────────────────────────────── */
.js-plotly-plot .plotly { border-radius: 14px; overflow: hidden; }
.element-container iframe { border: none !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
details, [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

/* ── Inputs & selects ────────────────────────────────────────────────────── */
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] label {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 0.85rem !important;
    min-height: 44px !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    min-height: 44px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(255,255,255,0.85) !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebarContent"] h1,
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3 {
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stRadio"] > label {
    font-family: 'Inter', sans-serif !important;
    padding: 10px 0 !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Animations ──────────────────────────────────────────────────────────── */
@keyframes slide-down {
    from { transform: translateY(-24px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}
@keyframes pulse-border {
    0%, 100% { border-left-color: #ff2244; opacity: 1; }
    50%       { border-left-color: #ff7744; opacity: 0.9; }
}
@keyframes fade-up {
    from { transform: translateY(16px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}
@keyframes throb {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,34,68,0.0); }
    50%       { box-shadow: 0 0 0 6px rgba(255,34,68,0.15); }
}

/* ── Alert banners ───────────────────────────────────────────────────────── */
.banner-unhealthy {
    animation: slide-down 0.45s cubic-bezier(0.4,0,0.2,1) both,
               pulse-border 2.5s ease-in-out 0.45s infinite,
               throb 2.5s ease-in-out 0.45s infinite;
}
.banner-hazardous {
    animation: slide-down 0.45s cubic-bezier(0.4,0,0.2,1) both;
}

/* ── Status bar ──────────────────────────────────────────────────────────── */
.status-bar {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Space Grotesk', monospace;
    font-size: 11px; color: rgba(255,255,255,0.35);
    letter-spacing: 0.08em; padding: 4px 0 12px;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #00ff88;
    animation: dot-pulse 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes dot-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px #00ff88; }
    50%       { opacity: 0.4; box-shadow: none; }
}

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center; padding: 48px 24px;
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    animation: fade-up 0.4s cubic-bezier(0.4,0,0.2,1);
}

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    h1 { font-size: 1.4rem !important; }
    .stColumn { min-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Gauge component ────────────────────────────────────────────────────────────
def render_gauge(aqi: float, pm25: float) -> None:
    lbl, color, _ = aqi_category(aqi)
    needle_rot = max(-90.0, min(90.0, (aqi / 500) * 180 - 90))

    segs = "".join(f'<path d="{p}" fill="{c}" opacity="0.92"/>' for p, c in _SEG_PATHS)

    # Tick marks at category boundaries (angles: 180,162,144,126,108,72,0)
    ticks = ""
    for ang in [180, 162, 144, 126, 108, 72, 0]:
        r = math.radians(ang)
        x1 = _CX + (_RO - 1) * math.cos(r);  y1 = _CY - (_RO - 1) * math.sin(r)
        x2 = _CX + (_RO + 9) * math.cos(r);  y2 = _CY - (_RO + 9) * math.sin(r)
        ticks += (f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                  f'x2="{x2:.1f}" y2="{y2:.1f}" '
                  f'stroke="rgba(255,255,255,0.45)" stroke-width="1.8" stroke-linecap="round"/>')

    # AQI label positions along outer arc
    label_data = [
        (171, "0"), (153, "50"), (135, "100"), (117, "150"),
        (90,  "200"), (36, "300"), (0, "500"),
    ]
    arc_labels = ""
    for ang, txt in label_data:
        r = math.radians(ang)
        lx = _CX + (_RO + 22) * math.cos(r)
        ly = _CY - (_RO + 22) * math.sin(r)
        arc_labels += (f'<text x="{lx:.1f}" y="{ly:.1f}" '
                       f'text-anchor="middle" dominant-baseline="middle" '
                       f'font-family="Space Grotesk, monospace" font-size="9" '
                       f'fill="rgba(255,255,255,0.35)">{txt}</text>')

    html = f"""<!DOCTYPE html><html><head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Syne:wght@800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:transparent;overflow:hidden;display:flex;justify-content:center}}
.wrap{{position:relative;width:360px;height:260px}}
.glow{{
  position:absolute;top:30px;left:50%;transform:translateX(-50%);
  width:290px;height:150px;border-radius:50% 50% 0 0/100% 100% 0 0;
  border:2px solid {color};border-bottom:none;
  box-shadow:0 0 18px {color}50,0 0 40px {color}25,inset 0 0 18px {color}12;
  animation:breathe 2.8s ease-in-out infinite;pointer-events:none;
}}
@keyframes breathe{{
  0%,100%{{opacity:.45;box-shadow:0 0 12px {color}40,0 0 28px {color}18,inset 0 0 12px {color}08}}
  50%    {{opacity:1;  box-shadow:0 0 32px {color}90,0 0 60px {color}45,inset 0 0 24px {color}20}}
}}
.needle{{
  transform-origin:{_CX}px {_CY}px;
  transform:rotate(-90deg);
  animation:settle 1.9s cubic-bezier(0.34,1.56,0.64,1) forwards;
}}
@keyframes settle{{
  0%   {{transform:rotate(-90deg)}}
  100% {{transform:rotate({needle_rot:.2f}deg)}}
}}
.center{{
  position:absolute;bottom:4px;left:0;right:0;
  text-align:center;animation:pop-in .5s cubic-bezier(.4,0,.2,1) 1.7s both
}}
@keyframes pop-in{{from{{opacity:0;transform:scale(.85)}}to{{opacity:1;transform:scale(1)}}}}
.n{{
  font-family:'Space Grotesk',monospace;font-size:76px;font-weight:700;
  color:{color};line-height:1;
  text-shadow:0 0 24px {color}70,0 0 50px {color}35;display:block
}}
.cat{{
  font-family:'Space Grotesk',sans-serif;font-size:11px;font-weight:500;
  color:rgba(255,255,255,.6);letter-spacing:.18em;text-transform:uppercase;
  display:block;margin-top:3px
}}
.live{{
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
  border-radius:20px;padding:3px 11px;margin-top:7px;
  font-family:'Space Grotesk',monospace;font-size:10px;font-weight:600;
  color:rgba(255,255,255,.8);letter-spacing:.12em
}}
.dot{{width:6px;height:6px;border-radius:50%;background:#00ff88;animation:blink 1.3s ease-in-out infinite}}
@keyframes blink{{0%,100%{{opacity:1;box-shadow:0 0 6px #00ff88}}50%{{opacity:.15;box-shadow:none}}}}
</style></head><body>
<div class="wrap">
  <div class="glow" aria-hidden="true"></div>
  <svg viewBox="0 0 320 170" width="360" height="170" style="display:block">
    <!-- Background ring -->
    <path d="{_arc(_CX,_CY,_RO+3,_RI-3,180,0)}" fill="rgba(255,255,255,0.03)"/>
    <!-- AQI segments -->
    {segs}
    <!-- Tick marks -->
    {ticks}
    <!-- Arc labels -->
    {arc_labels}
    <!-- Needle shadow (blur) -->
    <line x1="{_CX}" y1="{_CY+10}" x2="{_CX}" y2="{_CY-115}"
          stroke="rgba(0,0,0,.55)" stroke-width="6" stroke-linecap="round"
          class="needle" style="filter:blur(3px)"/>
    <!-- Needle -->
    <line x1="{_CX}" y1="{_CY+12}" x2="{_CX}" y2="{_CY-118}"
          stroke="white" stroke-width="2.5" stroke-linecap="round"
          class="needle" role="img" aria-label="AQI needle at {aqi:.0f}"/>
    <!-- Hub -->
    <circle cx="{_CX}" cy="{_CY}" r="11" fill="{color}" opacity=".95"/>
    <circle cx="{_CX}" cy="{_CY}" r="5.5" fill="#0a0e1a"/>
  </svg>
  <div class="center" role="status" aria-live="polite">
    <span class="n" aria-label="AQI {aqi:.0f}">{aqi:.0f}</span>
    <span class="cat">{lbl}</span>
    <span class="live" aria-label="Live reading">
      <span class="dot" aria-hidden="true"></span>LIVE
    </span>
  </div>
</div>
</body></html>"""
    components.html(html, height=268, scrolling=False)


# ── Alert banners ──────────────────────────────────────────────────────────────
def render_alert(aqi: float, pm25: float) -> None:
    to_addr = os.getenv("ALERT_TO", os.getenv("ALERT_EMAIL", "—"))

    if aqi > 200:
        email_sent = send_alert(aqi, pm25)
        email_log = ""
        if os.getenv("ALERT_EMAIL"):
            status = "✓ Sent" if email_sent else "✗ Failed"
            email_log = (
                f'<div style="background:rgba(0,0,0,.45);border:1px solid rgba(255,34,68,.25);'
                f'border-radius:8px;padding:10px 14px;font-family:\'Space Grotesk\',monospace;'
                f'font-size:11px;color:#00ff88;margin-top:12px;text-align:left">'
                f'<span style="opacity:.5">► </span>Preparing email alert...<br>'
                f'<span style="opacity:.5">► </span>Sending to {to_addr}<br>'
                f'<span style="color:{"#00ff88" if email_sent else "#ff2244"}">► {status}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="banner-hazardous" role="alert" aria-live="assertive" '
            f'style="background:linear-gradient(135deg,#3d0010,#7e0023);'
            f'color:#fff;padding:22px 24px;border-radius:14px;'
            f'border:1px solid rgba(255,34,68,.45);margin-bottom:16px;'
            f'box-shadow:0 0 40px rgba(255,34,68,.25),0 0 80px rgba(255,34,68,.1)">'
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<span style="font-size:28px" aria-hidden="true">🚨</span>'
            f'<div>'
            f'<h3 style="margin:0;font-family:Syne,sans-serif;font-size:1.1rem;'
            f'letter-spacing:.05em">HAZARDOUS AIR QUALITY — AQI {aqi:.0f}</h3>'
            f'<p style="margin:6px 0 0;opacity:.85;font-size:.9rem;line-height:1.5">'
            f'<strong>Health Advisory:</strong> Avoid ALL outdoor activity. Everyone is at serious risk. '
            f'Wear N95/P100 masks if outside. Keep windows and doors closed. Use air purifiers.'
            f'</p>'
            f'<p style="margin:6px 0 0;opacity:.65;font-size:.8rem">'
            f'PM2.5: {pm25:.1f} µg/m³ — WHO 24h limit: 15 µg/m³ ({pm25/15:.1f}× exceeded)'
            f'</p>'
            f'</div></div>{email_log}</div>',
            unsafe_allow_html=True,
        )

    elif aqi > 150:
        st.markdown(
            f'<div class="banner-unhealthy" role="alert" aria-live="polite" '
            f'style="background:rgba(255,34,68,.12);color:#fff;padding:18px 24px;'
            f'border-radius:14px;border-left:5px solid #ff2244;'
            f'border-top:1px solid rgba(255,34,68,.2);'
            f'border-right:1px solid rgba(255,34,68,.1);'
            f'border-bottom:1px solid rgba(255,34,68,.1);margin-bottom:16px">'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:22px" aria-hidden="true">⚠️</span>'
            f'<div>'
            f'<h3 style="margin:0;font-family:Syne,sans-serif;font-size:1rem">'
            f'UNHEALTHY AIR QUALITY — AQI {aqi:.0f}</h3>'
            f'<p style="margin:5px 0 0;opacity:.8;font-size:.85rem;line-height:1.5">'
            f'<strong>Health Advisory:</strong> Sensitive groups (children, elderly, respiratory conditions) '
            f'should avoid prolonged outdoor exertion. Everyone should reduce outdoor activity. '
            f'Wear a mask if going outside.'
            f'</p>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )


# ── Status bar ─────────────────────────────────────────────────────────────────
def render_status(ts: datetime, label: str = "Live data") -> None:
    st.markdown(
        f'<div class="status-bar" role="status" aria-label="{label}">'
        f'<span class="status-dot" aria-hidden="true"></span>'
        f'{label} &nbsp;·&nbsp; {ts.strftime("%Y-%m-%d %H:%M %Z")}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Empty state ────────────────────────────────────────────────────────────────
def render_empty(icon: str, title: str, msg: str) -> None:
    st.markdown(
        f'<div class="empty-state" role="alert">'
        f'<div style="font-size:48px;margin-bottom:12px" aria-hidden="true">{icon}</div>'
        f'<h3 style="font-family:Syne,sans-serif;margin-bottom:8px">{title}</h3>'
        f'<p style="opacity:.6;font-size:.9rem;max-width:360px;margin:0 auto">{msg}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Charts ─────────────────────────────────────────────────────────────────────
def _forecast_chart(forecasts: list[dict]) -> go.Figure:
    df = pd.DataFrame(forecasts)
    fig = go.Figure()

    # Gradient confidence band — stacked fills for depth
    for alpha, scale in [(0.06, 1.3), (0.1, 1.15), (0.15, 1.0)]:
        fig.add_trace(go.Scatter(
            x=list(df["date"]) + list(df["date"])[::-1],
            y=list(df["aqi"] * scale) + list(df["aqi"] / scale)[::-1],
            fill="toself",
            fillcolor=f"rgba(99,102,241,{alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))

    # Confidence band (actual)
    fig.add_trace(go.Scatter(
        x=list(df["date"]) + list(df["date"])[::-1],
        y=list(df["aqi_hi"]) + list(df["aqi_lo"])[::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty band", hoverinfo="skip",
    ))

    # Neon forecast line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["aqi"],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Colored markers with rich tooltips
    for _, row in df.iterrows():
        lung = "🫁" if row["aqi"] <= 100 else ("😷" if row["aqi"] <= 200 else "🚨")
        fig.add_trace(go.Scatter(
            x=[row["date"]], y=[row["aqi"]],
            mode="markers+text",
            marker=dict(color=row["color"], size=18,
                        line=dict(color="white", width=2),
                        symbol="circle"),
            text=[f'{row["aqi"]:.0f}'],
            textposition="top center",
            textfont=dict(color="white", size=12, family="Space Grotesk"),
            name=row["label"],
            customdata=[[row["pm25"], row["pm10"], row["aqi_lo"],
                         row["aqi_hi"], row["label"], lung]],
            hovertemplate=(
                f"<b>{row['date']}</b><br>"
                "%{customdata[5]} AQI: <b>%{customdata[0]:.0f}</b><br>"
                "Category: <b>%{customdata[4]}</b><br>"
                "PM2.5: %{customdata[0]:.1f} µg/m³<br>"
                "PM10: %{customdata[1]:.1f} µg/m³<br>"
                "Range: %{customdata[2]:.0f} – %{customdata[3]:.0f}"
                "<extra></extra>"
            ),
        ))

    # Day separator lines
    for date in df["date"]:
        fig.add_vline(x=date, line_dash="dot",
                      line_color="rgba(255,255,255,0.08)", line_width=1)

    # WHO threshold lines
    for thresh, lbl, clr in [
        (50,  "Good",          "#00ff88"),
        (100, "Moderate",      "#ffb347"),
        (150, "USG",           "#ff6b35"),
        (200, "Unhealthy",     "#ff2244"),
        (300, "Very Unhealthy","#9b59b6"),
    ]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=clr, opacity=0.4,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font=dict(color=clr, size=10,
                                           family="Space Grotesk, monospace"))

    max_y = max(df["aqi_hi"].max(), 220)
    layout = dict(**_PLOT,
                  height=420,
                  xaxis_title="Date", yaxis_title="AQI",
                  yaxis_range=[0, max_y + 40],
                  legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                  title=dict(text="3-Day AQI Forecast",
                             font=dict(family="Syne", size=15, color="rgba(255,255,255,.8)")))
    fig.update_layout(**layout)
    return fig


def _historical_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Shaded AQI bands (very subtle)
    for lo, hi, lbl, bg, _ in AQI_CATS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=bg, opacity=0.04,
                      annotation_text=lbl,
                      annotation_position="right",
                      annotation_font=dict(size=9, family="Space Grotesk, monospace",
                                           color=bg))

    # AQI line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["aqi"],
        mode="lines", name="AQI",
        line=dict(color="rgba(255,255,255,.55)", width=1.2),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>AQI: <b>%{y}</b><extra></extra>",
    ))

    # Color-coded markers
    for lo, hi, lbl, bg, _ in AQI_CATS:
        mask = (df["aqi"] >= lo) & (df["aqi"] <= hi)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "date"], y=df.loc[mask, "aqi"],
                mode="markers", name=lbl,
                marker=dict(color=bg, size=3.5, opacity=0.85),
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>AQI: <b>%{{y}}</b> — {lbl}<extra></extra>",
            ))

    # PM2.5 secondary axis
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pm2_5"],
        name="PM2.5 (µg/m³)", yaxis="y2",
        line=dict(color="rgba(0,180,216,.55)", width=1, dash="dot"),
        hovertemplate="PM2.5: <b>%{y:.1f} µg/m³</b><extra></extra>",
    ))

    layout = dict(**_PLOT,
                  height=460,
                  xaxis=dict(
                      **_PLOT["xaxis"],
                      rangeslider=dict(
                          visible=True,
                          bgcolor="rgba(255,255,255,.02)",
                          bordercolor="rgba(255,255,255,.07)",
                          thickness=0.06,
                      ),
                      type="date",
                      title_text="Date",
                  ),
                  yaxis=dict(**_PLOT["yaxis"], title_text="AQI"),
                  yaxis2=dict(
                      title="PM2.5 (µg/m³)", overlaying="y", side="right",
                      gridcolor="rgba(0,0,0,0)",
                      tickfont=dict(family="Space Grotesk, monospace", size=9),
                      title_font=dict(size=10),
                  ),
                  legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                  title=dict(text="Historical AQI — Karachi",
                             font=dict(family="Syne", size=14, color="rgba(255,255,255,.8)")))
    fig.update_layout(**layout)
    return fig


def _importance_chart(model, fcols: list) -> go.Figure | None:
    try:
        names = model.get_booster().feature_names or fcols
        imps  = model.feature_importances_
        fi = (pd.DataFrame({"feature": names[:len(imps)], "importance": imps})
                .sort_values("importance", ascending=True).tail(15))
        fi["icon"]  = fi["feature"].map(lambda x: FEATURE_ICONS.get(x, "📊"))
        fi["label"] = fi.apply(lambda r: f'{r["icon"]} {r["feature"]}', axis=1)

        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["label"],
            orientation="h",
            marker=dict(
                color=fi["importance"],
                colorscale=[[0, "#0a1628"], [0.3, "#1a3a6b"],
                            [0.6, "#4a6fa5"], [1.0, "#00ff88"]],
                showscale=False,
                line=dict(color="rgba(255,255,255,.08)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            text=[f"{v:.3f}" for v in fi["importance"]],
            textposition="outside",
            textfont=dict(size=9, family="Space Grotesk, monospace",
                          color="rgba(255,255,255,.45)"),
        ))
        layout = dict(**_PLOT,
                      height=480,
                      margin=dict(t=40, b=10, l=10, r=60),
                      xaxis_title="Importance",
                      title=dict(text="Feature Importance — PM2.5 Model",
                                 font=dict(family="Syne", size=13,
                                           color="rgba(255,255,255,.8)")))
        fig.update_layout(**layout)
        return fig
    except Exception:
        return None


def _heatmap_chart(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["month_label"] = df2["date"].dt.to_period("M").astype(str)
    df2["day"]         = df2["date"].dt.day
    pivot = df2.pivot_table(index="month_label", columns="day", values="aqi", aggfunc="mean")

    scale = [[0.0, "#00ff8830"], [0.1, "#ffb34750"], [0.2, "#ff6b3560"],
             [0.35,"#ff224470"], [0.55,"#9b59b670"], [1.0, "#7e002380"]]
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=scale,
                    title="Daily AQI Calendar Heatmap")
    layout = dict(**_PLOT,
                  height=360,
                  coloraxis_colorbar=dict(
                      title="AQI",
                      tickfont=dict(family="Space Grotesk, monospace", size=9),
                      thickness=12,
                  ))
    fig.update_layout(**layout)
    return fig


# ── Pages ──────────────────────────────────────────────────────────────────────
def page_realtime(m25, m10, scaler, fs, fcols):
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)

    st.title("Real-Time AQI Prediction")
    render_status(now, "Live data — Karachi, Pakistan")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        refreshed = st.button("↺  Refresh", help="Fetch latest data from API",
                              use_container_width=True)

    with st.spinner("Fetching live atmospheric data…"):
        row = build_row(now, fs, scaler)

    if row is None:
        render_empty(
            "📡", "API Unavailable",
            "Could not reach Open-Meteo. Using last stored feature row for prediction. "
            "Check your connection and try refreshing.",
        )
        row = fs.iloc[-1]
        st.button("Retry", key="retry_rt")

    pm25, pm10 = predict(row, fcols, m25, m10)
    aqi        = pm25_to_aqi(pm25)
    lbl, color, _ = aqi_category(aqi)

    render_alert(aqi, pm25)

    # ── Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AQI (PM2.5-based)", f"{aqi:.0f}", lbl)
    c2.metric("PM2.5 Concentration", f"{pm25:.1f} µg/m³",
              help="Particulate matter < 2.5 µm — primary health indicator")
    c3.metric("PM10 Concentration", f"{pm10:.1f} µg/m³",
              help="Particulate matter < 10 µm")
    ratio = pm25 / 15.0
    c4.metric("vs WHO 24h Limit", f"{ratio:.1f}×",
              "✓ Within safe limit" if pm25 <= 15 else f"✗ {pm25-15:.1f} µg/m³ over",
              delta_color="inverse",
              help="WHO 24h PM2.5 guideline = 15 µg/m³")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + PM2.5 gauge
    g1, g2 = st.columns(2)
    with g1:
        render_gauge(aqi, pm25)
    with g2:
        # PM2.5 plotly gauge
        fig_pm = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pm25,
            number=dict(suffix=" µg/m³", font=dict(
                family="Space Grotesk, monospace", size=36)),
            delta=dict(reference=15, relative=False,
                       increasing=dict(color="#ff2244"),
                       decreasing=dict(color="#00ff88")),
            title=dict(text="PM2.5  ·  µg/m³",
                       font=dict(family="Syne, sans-serif", size=14,
                                 color="rgba(255,255,255,.7)")),
            gauge=dict(
                axis=dict(range=[0, 250], tickfont=dict(
                    family="Space Grotesk, monospace", size=9)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  12],  color="rgba(0,255,136,.08)"),
                    dict(range=[12, 35.4],color="rgba(255,179,71,.08)"),
                    dict(range=[35.4,55.4],color="rgba(255,107,53,.08)"),
                    dict(range=[55.4,150.4],color="rgba(255,34,68,.08)"),
                    dict(range=[150.4,250],color="rgba(155,89,182,.08)"),
                ],
                threshold=dict(
                    line=dict(color="#ffb347", width=2),
                    thickness=0.8, value=15,
                ),
            ),
        ))
        fig_pm.update_layout(
            height=268,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, monospace", color="rgba(255,255,255,.7)"),
            margin=dict(t=50, b=10, l=20, r=20),
            annotations=[dict(x=0.5, y=0.08, xref="paper", yref="paper",
                              text="← WHO limit: 15 µg/m³",
                              showarrow=False,
                              font=dict(size=9, color="#ffb347",
                                        family="Space Grotesk, monospace"))],
        )
        st.plotly_chart(fig_pm, use_container_width=True,
                        config=dict(displayModeBar=False))

    # ── AQI legend
    with st.expander("AQI Category Reference", expanded=False):
        leg = st.columns(len(AQI_CATS))
        for (lo, hi, lbl, bg, tc), col in zip(AQI_CATS, leg):
            col.markdown(
                f'<div style="background:{bg};color:{tc};padding:10px 6px;'
                f'border-radius:8px;text-align:center;font-size:.72rem;'
                f'font-family:Space Grotesk,monospace;line-height:1.35">'
                f'<b>{lo}–{hi}</b><br><span style="font-size:.65rem">{lbl}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Meteorological conditions
    with st.expander("Current Meteorological Conditions", expanded=False):
        met_map = {
            "temperature_2m": ("🌡️", "Temperature", "°C"),
            "relative_humidity_2m": ("💦", "Humidity", "%"),
            "wind_speed_10m": ("💨", "Wind Speed", "km/h"),
            "precipitation": ("🌧️", "Precipitation", "mm"),
            "cloud_cover": ("☁️", "Cloud Cover", "%"),
            "surface_pressure": ("🔵", "Pressure", "hPa"),
            "carbon_monoxide": ("🏭", "CO", "µg/m³"),
            "nitrogen_dioxide": ("⚗️", "NO₂", "µg/m³"),
        }
        avail = [(v, row[k]) for k, v in met_map.items() if k in row.index]
        mc = st.columns(4)
        for i, ((ico, lbl, unit), val) in enumerate(avail):
            mc[i % 4].metric(f"{ico} {lbl}", f"{float(val):.1f} {unit}")


def page_forecast(m25, m10, scaler, fs, fcols):
    tz    = pytz.timezone(TIMEZONE)
    today = datetime.now(tz).date()

    st.title("3-Day AQI Forecast")
    render_status(datetime.now(tz), "Forecast generated")

    forecasts = []
    hist      = fs.copy()
    api_failed = False

    with st.spinner("Generating 3-day ensemble forecast…"):
        for offset in range(1, 4):
            dt = datetime.combine(
                today + timedelta(days=offset), datetime.min.time()
            ).replace(tzinfo=tz)
            row = build_row(dt, hist, scaler)

            if row is None:
                api_failed = True
                row = hist.iloc[-1].copy()
                ts = pd.Timestamp(dt.date()); m = ts.month
                row.update(dict(
                    month=m, day_of_week=ts.dayofweek,
                    is_weekend=int(ts.dayofweek >= 5),
                    day_of_year=ts.dayofyear,
                    season=1 if m in [12,1,2] else (2 if m in [3,4,5] else (3 if m in [6,7,8] else 4)),
                ))

            pm25_p, pm10_p = predict(row, fcols, m25, m10)
            aqi = pm25_to_aqi(pm25_p)
            lbl, color, _ = aqi_category(aqi)
            unc = 0.10 + 0.05 * offset  # uncertainty grows with horizon
            forecasts.append(dict(
                date=dt.strftime("%a %b %d"),
                pm25=pm25_p, pm10=pm10_p, aqi=aqi,
                aqi_lo=pm25_to_aqi(max(0, pm25_p * (1 - unc))),
                aqi_hi=pm25_to_aqi(pm25_p * (1 + unc)),
                label=lbl, color=color,
            ))
            # propagate prediction as lag for next iteration
            syn = {"date": pd.Timestamp(dt.date()), "pm2_5": pm25_p, "pm10": pm10_p}
            syn.update({c: row[c] for c in hist.columns if c not in syn})
            hist = pd.concat([hist, pd.DataFrame([syn])], ignore_index=True)

    if api_failed:
        render_empty(
            "📡", "Partial API Data",
            "Some forecast days used last-known conditions. Predictions may be less accurate.",
        )
        st.button("Retry", key="retry_fc")

    # ── Forecast cards
    st.subheader("Daily Outlook")
    for f in forecasts:
        cc, cg, cs = st.columns([1, 2, 2])
        with cc:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{f["color"]}22,{f["color"]}44);'
                f'border:1px solid {f["color"]}55;padding:20px 16px;border-radius:14px;'
                f'text-align:center;height:100%;backdrop-filter:blur(12px);'
                f'animation:fade-up .5s cubic-bezier(.4,0,.2,1) both">'
                f'<div style="font-family:Space Grotesk,monospace;font-size:.75rem;'
                f'letter-spacing:.15em;text-transform:uppercase;opacity:.6;margin-bottom:6px">'
                f'{f["date"]}</div>'
                f'<div style="font-family:Space Grotesk,monospace;font-size:3.2rem;'
                f'font-weight:700;color:{f["color"]};line-height:1;'
                f'text-shadow:0 0 20px {f["color"]}60">{f["aqi"]:.0f}</div>'
                f'<div style="font-size:.72rem;opacity:.7;margin-top:6px">{f["label"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with cg:
            render_gauge(f["aqi"], f["pm25"])
        with cs:
            st.metric("PM2.5", f'{f["pm25"]:.1f} µg/m³')
            st.metric("PM10",  f'{f["pm10"]:.1f} µg/m³')
            st.metric("AQI Range", f'{f["aqi_lo"]:.0f} – {f["aqi_hi"]:.0f}',
                      help=f"±{(0.10+0.05*forecasts.index(f))*100:.0f}% uncertainty band")
        st.divider()

    # ── Main forecast chart
    st.plotly_chart(_forecast_chart(forecasts),
                    use_container_width=True,
                    config=dict(displayModeBar=False))

    # ── Forecast alert
    worst_aqi  = max(f["aqi"]  for f in forecasts)
    worst_pm25 = max(f["pm25"] for f in forecasts)
    if worst_aqi > 150:
        st.markdown("---")
        render_alert(worst_aqi, worst_pm25)


def page_historical(m25, fs, fcols):
    st.title("Historical AQI Analysis")

    min_d, max_d = fs["date"].min().date(), fs["date"].max().date()
    default_start = max(min_d, max_d - timedelta(days=90))

    dc1, dc2 = st.columns(2)
    start = dc1.date_input("From", value=default_start,
                           min_value=min_d, max_value=max_d,
                           help="Analysis start date")
    end   = dc2.date_input("To",   value=max_d,
                           min_value=min_d, max_value=max_d,
                           help="Analysis end date")

    df = fs[(fs["date"].dt.date >= start) & (fs["date"].dt.date <= end)].copy()
    if df.empty:
        render_empty("🗓️", "No Data", "No records in the selected date range. Try a wider window.")
        return

    df["aqi"]      = df["pm2_5"].apply(pm25_to_aqi)
    df["category"] = df["aqi"].apply(lambda x: aqi_category(x)[0])

    render_status(
        datetime.now(pytz.timezone(TIMEZONE)),
        f"{len(df)} days loaded · {start} → {end}",
    )

    # ── Summary KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean AQI",        f"{df['aqi'].mean():.0f}")
    k2.metric("Max AQI",         f"{df['aqi'].max():.0f}")
    k3.metric("Mean PM2.5",      f"{df['pm2_5'].mean():.1f} µg/m³")
    k4.metric("Days Unhealthy",  f"{(df['aqi'] > 150).sum()}",
              help="Days with AQI > 150")
    k5.metric("Days Hazardous",  f"{(df['aqi'] > 300).sum()}",
              help="Days with AQI > 300")

    st.divider()

    # ── Time series (full width) with rangeslider
    st.plotly_chart(_historical_chart(df),
                    use_container_width=True,
                    config=dict(displayModeBar=True,
                                modeBarButtonsToRemove=["select2d","lasso2d"],
                                displaylogo=False))

    # ── PM10 expander
    with st.expander("PM10 Trend", expanded=False):
        fig10 = go.Figure(go.Scatter(
            x=df["date"], y=df["pm10"],
            mode="lines",
            line=dict(color="rgba(247,127,0,.7)", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(247,127,0,.06)",
            hovertemplate="<b>%{x|%b %d}</b><br>PM10: %{y:.1f} µg/m³<extra></extra>",
        ))
        who_pm10 = 45  # WHO 24h PM10 guideline
        fig10.add_hline(y=who_pm10, line_dash="dot",
                        line_color="#ffb347", opacity=0.5,
                        annotation_text="WHO limit (45 µg/m³)",
                        annotation_font=dict(color="#ffb347", size=10,
                                             family="Space Grotesk, monospace"))
        fig10.update_layout(**dict(_PLOT,
                                   height=240,
                                   yaxis_title="PM10 (µg/m³)",
                                   title=dict(text="PM10 Concentration",
                                              font=dict(family="Syne", size=13))))
        st.plotly_chart(fig10, use_container_width=True,
                        config=dict(displayModeBar=False))

    st.divider()

    # ── Feature importance (left) + Category pie (right)
    fi_col, pie_col = st.columns([3, 2])
    with fi_col:
        fig_fi = _importance_chart(m25, fcols)
        if fig_fi:
            st.plotly_chart(fig_fi, use_container_width=True,
                            config=dict(displayModeBar=False))
        else:
            st.info("Feature importance not available for this model.")

    with pie_col:
        st.subheader("Category Distribution")
        cat_map  = {lbl: bg for _, _, lbl, bg, _ in AQI_CATS}
        cat_cnt  = df["category"].value_counts().reset_index()
        cat_cnt.columns = ["category", "days"]
        fig_pie = px.pie(
            cat_cnt, names="category", values="days",
            color="category", color_discrete_map=cat_map,
            hole=0.45,
        )
        fig_pie.update_traces(
            textfont=dict(family="Space Grotesk, monospace", size=10),
            marker=dict(line=dict(color="#050810", width=2)),
        )
        fig_pie.update_layout(
            **dict(_PLOT, height=380, margin=dict(t=10,b=10,l=10,r=10),
                   showlegend=True,
                   legend=dict(font=dict(size=9), orientation="v")),
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config=dict(displayModeBar=False))

    # ── Monthly heatmap
    st.divider()
    st.subheader("Calendar Heatmap")
    st.plotly_chart(_heatmap_chart(df), use_container_width=True,
                    config=dict(displayModeBar=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AQI Intelligence — Karachi",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    try:
        m25, m10, scaler = load_models()
        fs    = load_feature_store()
        fcols = feature_cols(fs)
    except FileNotFoundError as e:
        render_empty(
            "🔧", "Pipeline Not Ready",
            f"Required file not found: {e}. Run the training pipeline first, then reload.",
        )
        st.stop()

    # ── Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;'
            'letter-spacing:-.01em;margin-bottom:2px">🌫️ AQI Intelligence</div>'
            '<div style="font-family:Space Grotesk,monospace;font-size:.7rem;'
            'letter-spacing:.15em;opacity:.45;text-transform:uppercase;'
            'margin-bottom:16px">Climate Monitor · Karachi</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family:Space Grotesk,monospace;font-size:.75rem;'
            f'opacity:.5;padding:8px 0 4px">'
            f'{LAT}°N &nbsp; {LON}°E &nbsp; · &nbsp; {TIMEZONE}</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠  Real-time", "📈  Forecast", "📅  Historical"],
            label_visibility="collapsed",
        )

        st.divider()

        if os.getenv("ALERT_EMAIL"):
            st.markdown(
                '<div style="background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.2);'
                'border-radius:8px;padding:10px 12px;font-size:.75rem;'
                'font-family:Space Grotesk,monospace;color:rgba(0,255,136,.9)">'
                '📧 Email alerts: ARMED</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.expander("📧 Email alerts: OFF"):
                st.caption(
                    "Set env vars to arm:\n"
                    "- `ALERT_EMAIL`\n"
                    "- `ALERT_PASSWORD`\n"
                    "- `ALERT_TO`"
                )

        st.divider()
        st.markdown(
            f'<div style="font-family:Space Grotesk,monospace;font-size:.68rem;'
            f'opacity:.35;line-height:1.8">'
            f'Model: XGBoost<br>'
            f'Features: {len(fcols)}<br>'
            f'History: {fs["date"].min().date()}<br>'
            f'→ {fs["date"].max().date()}<br>'
            f'Source: Open-Meteo API'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Route
    if "Real-time" in page:
        page_realtime(m25, m10, scaler, fs, fcols)
    elif "Forecast" in page:
        page_forecast(m25, m10, scaler, fs, fcols)
    else:
        page_historical(m25, fs, fcols)


if __name__ == "__main__":
    main()
