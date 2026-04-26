# Phase 3: Data Cleaning & Preprocessing

import sys
sys.path.append('../')

from src.preprocess import run_pipeline
import pandas as pd

# Run full pipeline
df = run_pipeline()

# Quick check
print("\nSample:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
