"""Project paths and default hyperparameters relative to the repository root."""

from __future__ import annotations

from pathlib import Path

# Repository root (parent of the seed/ package)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CATALOG_DIR = DATA_DIR / "catalogs"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
FAULTS_DIR = DATA_DIR / "faults"
MODELS_DIR = ROOT / "models"
RESULTS_SAMPLES_DIR = ROOT / "results" / "samples"
OUTPUTS_DIR = ROOT / "outputs"

# Default sample inputs shipped with the repository
DEFAULT_EVENT = "jiuzhaigou"
DEFAULT_TEST_FEATURES = SAMPLES_DIR / "test_jiuzhaigou.csv"
DEFAULT_CATALOG_2000_2013 = CATALOG_DIR / "2000-2013.csv"
DEFAULT_CATALOG_2014_2025 = CATALOG_DIR / "2014-2025.csv"
DEFAULT_FAULT_SHAPEFILE = FAULTS_DIR / "active_faults_study_region.shp"

# Model / training defaults (match the paper experiments)
SEED = 1234
WINDOW_SIZE = 30
INPUT_SIZE = 6
HIDDEN_SIZE = 256
NUM_EPOCHS = 500
LEARNING_RATE = 0.003
FEATURE_NAMES = [
    "std_depthEQ",
    "std_intertime",
    "std_lat",
    "std_lon",
    "std_magnitude",
    "std_energy_release",
]

# Spatially varying prior probabilities keyed by event / region label
P0_DICT = {
    1: 0.0289,
    2: 0.1003,
    3: 0.1003,
    4: 0.1003,
    5: 0.1003,
    6: 0.1003,
    7: 0.0625,
    8: 0.0625,
    9: 0.0625,
    10: 0.0625,
    11: 0.1003,
    12: 0.0625,
    13: 0.0625,
    14: 0.0289,
    15: 0.0289,
    16: 0.01,
}

# Case-study events: each has a test-feature CSV and a model ensemble directory.
# predict.py auto-discovers all model_*_event_*.pth + RF_*.joblib pairs in model_dir.
EVENTS = {
    "jiuzhaigou": {
        "display_name": "Jiuzhaigou",
        "test_features": SAMPLES_DIR / "test_jiuzhaigou.csv",
        "model_dir": MODELS_DIR / "jiuzhaigou",
        "hypocenter": (103.82, 33.2),
        "post_event_days": 200,
    },
    "wenchuan": {
        "display_name": "Wenchuan",
        "test_features": SAMPLES_DIR / "test_wenchuan.csv",
        "model_dir": MODELS_DIR / "wenchuan",
        "hypocenter": (103.4, 31.0),
        "post_event_days": 200,
    },
    "lushan": {
        "display_name": "Lushan (2013)",
        "test_features": SAMPLES_DIR / "test_lushan.csv",
        "model_dir": MODELS_DIR / "lushan",
        "hypocenter": (102.9833, 30.3),
        "post_event_days": 200,
    },
    "lushan2022": {
        "display_name": "Lushan (2022)",
        "test_features": SAMPLES_DIR / "test_lushan2022.csv",
        "model_dir": MODELS_DIR / "lushan2022",
        "hypocenter": (102.94, 30.37),
        "post_event_days": 200,
    },
    "luding": {
        "display_name": "Luding",
        "test_features": SAMPLES_DIR / "test_luding.csv",
        "model_dir": MODELS_DIR / "luding",
        "hypocenter": (102.08, 29.59),
        "post_event_days": 200,
    },
    "jishishan": {
        "display_name": "Jishishan",
        "test_features": SAMPLES_DIR / "test_jishishan.csv",
        "model_dir": MODELS_DIR / "jishishan",
        "hypocenter": (102.79, 35.7),
        "post_event_days": 200,
    },
}


def get_event_config(event: str) -> dict:
    """Return config for a named event (case-insensitive)."""
    key = event.strip().lower()
    if key not in EVENTS:
        known = ", ".join(sorted(EVENTS))
        raise KeyError(f"Unknown event '{event}'. Choose one of: {known}")
    return EVENTS[key]


def ensure_output_dirs() -> None:
    """Create directories that store generated artifacts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
