# Step_1_fix_radar
import pandas as pd
import os
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

EXPECTED_COLS = 1135  # timestamp + 1134 radar features

# ----------------------------------------------------------
# Paths are resolved relative to THIS script's location, so it
# works regardless of the directory you launch it from. Expected:
#
#   Data_Augmentation/
#   ├── Dataset/
#   │   ├── subject_1/  radar.csv
#   │   ├── subject_2/  radar.csv
#   │   └── subject_10/ radar.csv
#   └── Stage_1_Data_preparation/
#       └── fix_radar.py   <-- this file
#
# Run this BEFORE extraction_script.py: it rectangularizes each
# raw radar.csv (variable-length rows) into a fixed EXPECTED_COLS
# width, overwriting the file in place.
# ----------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent      # Stage_1_Data_preparation
PROJECT_ROOT = SCRIPT_DIR.parent                    # Data_Augmentation
DATASET_DIR  = PROJECT_ROOT / "Dataset"             # where subject_* folders live

RADAR_FILENAME = "radar.csv"                        # radar file inside each subject folder

# ==========================================================
# DISCOVER SUBJECT FOLDERS
# ==========================================================

subject_dirs = sorted(
    (p for p in DATASET_DIR.iterdir()
     if p.is_dir() and p.name.lower().startswith("subject_")),
    key=lambda p: int(p.name.split("_")[1])
)

print(f"Found {len(subject_dirs)} subject folders in {DATASET_DIR}")

# ==========================================================
# CLEAN ALL RADAR FILES
# ==========================================================

for subject_dir in subject_dirs:

    filepath = subject_dir / RADAR_FILENAME
    print(f"Processing {filepath} ...", end=" ")

    rows = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            tokens = line.split(",")

            try:
                values = [float(t) if t.strip() != "" else 0.0 for t in tokens]
            except ValueError:
                continue

            if len(values) == 0:
                continue

            # Zero-pad short rows
            if len(values) < EXPECTED_COLS:
                values += [0.0] * (EXPECTED_COLS - len(values))

            # Truncate overlong rows
            elif len(values) > EXPECTED_COLS:
                values = values[:EXPECTED_COLS]

            rows.append(values)

    df = pd.DataFrame(rows, columns=range(EXPECTED_COLS))

    # ----------------------------------------------------------
    # Overwrite original file with clean version
    # ----------------------------------------------------------

    df.to_csv(filepath, index=False, header=False)

    print(f"Done. Shape: {df.shape}")

print("\nAll radar files cleaned.")