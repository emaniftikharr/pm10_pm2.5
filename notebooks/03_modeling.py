# Phase 4: Model Training & Evaluation
# Models: Linear Regression, Random Forest, XGBoost

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys
sys.path.append('../')
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, TARGET_VARIABLES, RANDOM_STATE
