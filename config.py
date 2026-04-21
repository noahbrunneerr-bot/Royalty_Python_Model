from __future__ import annotations
from pathlib import Path

# Projekt Root = Ordner wo diese Datei liegt
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

GROUND_TRUTH_CSV = DATA_PROCESSED / "ground_truth_clean.csv"

VALID_SCENARIOS = {"Base", "Flat", "Downside"}

def ensure_dirs() -> None:
    for p in [DATA_RAW, DATA_PROCESSED, OUTPUTS_DIR, REPORTS_DIR, SRC_DIR]:
        p.mkdir(parents=True, exist_ok=True)
