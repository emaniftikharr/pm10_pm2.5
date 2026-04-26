from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
APP_DIR = ROOT_DIR / "app"

TARGET_VARIABLES = ["AQI", "PM2.5", "PM10"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
