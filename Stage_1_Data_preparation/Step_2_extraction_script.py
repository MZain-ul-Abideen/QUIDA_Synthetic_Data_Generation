# Step_2_extraction_script
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

WINDOW_BEFORE = 3  # seconds
WINDOW_AFTER = 3   # seconds

# ----------------------------------------------------------
# Paths are resolved relative to THIS script's location, so
# the script works no matter which directory you launch it
# from. Expected layout:
#
#   Data_Augmentation/
#   ├── Dataset/
#   │   ├── falls.csv
#   │   ├── subject_1/  {accelerometer,ir_camera,radar,LIDAR}.csv
#   │   ├── subject_2/  ...
#   │   └── subject_10/ ...
#   └── Stage_1_Data_preparation/
#       └── extraction_script.py   <-- this file
# ----------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent      # Stage_1_Data_preparation
PROJECT_ROOT = SCRIPT_DIR.parent                    # Data_Augmentation
DATASET_DIR  = PROJECT_ROOT / "Dataset"             # where subject_* folders live
OUTPUT_DIR   = SCRIPT_DIR / "processed"             # where extracted falls are written

# ----------------------------------------------------------
# Sensor map: output-name -> filename inside each subject folder.
# Output CSVs and metadata row-counts use the KEY (imu/ir/radar),
# exactly matching the original script. To also extract LIDAR,
# uncomment its line below.
# ----------------------------------------------------------

SENSORS = {
    "imu":   "accelerometer.csv",
    "ir":    "ir_camera.csv",
    "radar": "radar.csv",
    # "lidar": "LIDAR.csv",   # <-- uncomment to also extract LIDAR
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# SENSOR FILE LOADER
# ==========================================================

def read_sensor_csv(path):
    """Read a sensor CSV with header=None, tolerant of ragged rows.

    Raw radar frames contain a variable number of detections, so each
    row can have a different column count (this is what fix_radar.py
    rectangularized in the old flat dataset). We find the widest row
    first, then read with a fixed set of column positions so pandas
    doesn't infer a too-narrow width and crash. Timestamp stays in
    column 0; shorter rows are padded with NaN on the right.
    """
    with open(path, "r") as fh:
        max_cols = max((line.count(",") + 1 for line in fh if line.strip()), default=1)
    return pd.read_csv(path, header=None, names=range(max_cols))

# ==========================================================
# LOAD FALL TIMES
# ==========================================================

falls_df = pd.read_csv(DATASET_DIR / "falls.csv", header=None)

fall_counter = 1

# ==========================================================
# DISCOVER SUBJECT FOLDERS
# ==========================================================

# Crawl every "subject_<N>" folder inside Dataset/, sorted by the
# numeric id (so subject_10 comes after subject_9, not after subject_1).
subject_dirs = sorted(
    (p for p in DATASET_DIR.iterdir()
     if p.is_dir() and p.name.lower().startswith("subject_")),
    key=lambda p: int(p.name.split("_")[1])
)

print(f"Found {len(subject_dirs)} subject folders in {DATASET_DIR}")

# ==========================================================
# PROCESS ALL SUBJECTS
# ==========================================================

for subject_dir in subject_dirs:

    subject_id = int(subject_dir.name.split("_")[1])

    print(f"\nProcessing Subject {subject_id}  ({subject_dir.name})")

    # ------------------------------------------------------
    # Load sensor files for this subject
    # ------------------------------------------------------

    sensor_data = {}
    for key, filename in SENSORS.items():
        sensor_data[key] = read_sensor_csv(subject_dir / filename)

    shapes = " | ".join(f"{k.upper()}={df.shape}" for k, df in sensor_data.items())
    print(f"  {shapes}")

    # ------------------------------------------------------
    # Get subject fall timestamps (column-wise, as before)
    # ------------------------------------------------------

    subject_falls = falls_df.iloc[:, subject_id - 1].values

    # ------------------------------------------------------
    # Extract each fall
    # ------------------------------------------------------

    for fall_number, fall_time in enumerate(subject_falls, start=1):

        start_time = fall_time - WINDOW_BEFORE
        end_time   = fall_time + WINDOW_AFTER

        # --------------------------------------------------
        # Extract sensor windows (rows whose timestamp in col 0
        # falls inside the window)
        # --------------------------------------------------

        windows = {}
        for key, df in sensor_data.items():
            windows[key] = df[(df[0] >= start_time) & (df[0] <= end_time)]

        # --------------------------------------------------
        # Create fall folder
        # --------------------------------------------------

        fall_folder = os.path.join(
            OUTPUT_DIR,
            f"fall_{fall_counter:03d}"
        )

        os.makedirs(fall_folder, exist_ok=True)

        # --------------------------------------------------
        # Save sensor data
        # --------------------------------------------------

        for key, window in windows.items():
            window.to_csv(
                os.path.join(fall_folder, f"{key}.csv"),
                index=False,
                header=False
            )

        # --------------------------------------------------
        # Save metadata
        # --------------------------------------------------

        metadata = {
            "fall_id": fall_counter,
            "subject_id": subject_id,
            "fall_number": fall_number,
            "fall_timestamp": int(fall_time),
            "window_before_sec": WINDOW_BEFORE,
            "window_after_sec": WINDOW_AFTER,
        }
        for key, window in windows.items():
            metadata[f"{key}_rows"] = len(window)

        with open(
            os.path.join(fall_folder, "metadata.json"),
            "w"
        ) as f:
            json.dump(metadata, f, indent=4)

        rows_summary = " ".join(f"{k.upper()}={len(w)}" for k, w in windows.items())
        print(f"  Fall {fall_counter:03d} | {rows_summary}")

        fall_counter += 1

print("\nDone.")
print(f"Total Falls Extracted: {fall_counter - 1}")