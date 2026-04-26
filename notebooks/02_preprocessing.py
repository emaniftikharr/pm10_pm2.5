# Phase 3: Data Preprocessing & Feature Engineering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, TARGET_VARIABLES, TEST_SIZE, RANDOM_STATE
