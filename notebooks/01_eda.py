# Phase 2: Exploratory Data Analysis
# Targets: AQI, PM2.5, PM10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

sns.set_theme(style='whitegrid')
