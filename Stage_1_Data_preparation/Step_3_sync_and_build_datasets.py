# Step_3_sync_and_build_datasets

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =====================================================
# CONFIG
# =====================================================

PROCESSED_DIR = "processed"
OUTPUT_DIR = "../Preprocessed_dataset"
TARGET_FRAMES = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# INTERPOLATION FUNCTION
# =====================================================

def resample_sensor(data, target_frames=32):
    """
    Remove timestamp (col 0), then linearly interpolate
    each feature column to exactly target_frames rows.
    """
    values   = data[:, 1:]
    n        = len(values)
    old_time = np.linspace(0, 1, n)
    new_time = np.linspace(0, 1, target_frames)

    resampled = np.zeros(
        (target_frames, values.shape[1]),
        dtype=np.float32
    )

    for col in range(values.shape[1]):
        f = interp1d(old_time, values[:, col], kind="linear")
        resampled[:, col] = f(new_time)

    return resampled

# =====================================================
# STORAGE
# =====================================================

X_imu    = []
X_imu_ir = []
X_imu_radar = []
X_all    = []

metadata_rows = []

# =====================================================
# PROCESS FALLS
# =====================================================

fall_folders = sorted([
    f for f in os.listdir(PROCESSED_DIR)
    if f.startswith("fall_")
])

for fall_folder in fall_folders:

    folder_path = os.path.join(PROCESSED_DIR, fall_folder)

    # --------------------------------------------------
    # Load sensor files — catch empty files gracefully
    # --------------------------------------------------

    try:
        imu   = pd.read_csv(os.path.join(folder_path, "imu.csv"),   header=None).values
        ir    = pd.read_csv(os.path.join(folder_path, "ir.csv"),    header=None).values
        radar = pd.read_csv(os.path.join(folder_path, "radar.csv"), header=None).values
    except pd.errors.EmptyDataError:
        print(f"Skipping {fall_folder}  [empty file]")
        continue

    # --------------------------------------------------
    # Skip bad samples
    # --------------------------------------------------

    if len(imu) == 0:
        print(f"Skipping {fall_folder}  [IMU empty]")
        continue

    if len(ir) < 2:
        print(f"Skipping {fall_folder}  [IR too short]")
        continue

    if len(radar) < 2:
        print(f"Skipping {fall_folder}  [Radar too short]")
        continue

    # --------------------------------------------------
    # Resample all sensors to TARGET_FRAMES
    # --------------------------------------------------

    imu_sync   = resample_sensor(imu,   TARGET_FRAMES)
    ir_sync    = resample_sensor(ir,    TARGET_FRAMES)
    radar_sync = resample_sensor(radar, TARGET_FRAMES)

    # --------------------------------------------------
    # Experiment A — IMU only (32, 3)
    # --------------------------------------------------

    X_imu.append(imu_sync)

    # --------------------------------------------------
    # Experiment B — IMU + IR
    # --------------------------------------------------

    X_imu_ir.append(
        np.concatenate([imu_sync, ir_sync], axis=1)
    )

    # --------------------------------------------------
    # Experiment C — IMU + Radar
    # --------------------------------------------------

    X_imu_radar.append(
        np.concatenate([imu_sync, radar_sync], axis=1)
    )

    # --------------------------------------------------
    # Experiment D — IMU + IR + Radar
    # --------------------------------------------------

    X_all.append(
        np.concatenate([imu_sync, ir_sync, radar_sync], axis=1)
    )
    
    # --------------------------------------------------
    # Metadata — pull subject_id and fall_number from
    # the per-fall metadata.json saved during extraction
    # --------------------------------------------------

    with open(os.path.join(folder_path, "metadata.json")) as f:
        meta = json.load(f)

    metadata_rows.append({
        "fall_id":      meta["fall_id"],
        "subject_id":   meta["subject_id"],
        "fall_number":  meta["fall_number"],
    })

    print(
        f"  {fall_folder} | "
        f"subject={meta['subject_id']:02d} "
        f"fall={meta['fall_number']:02d} | "
        f"IMU={imu_sync.shape} "
        f"IR={ir_sync.shape} "
        f"Radar={radar_sync.shape}"
    )

# =====================================================
# CONVERT TO ARRAYS
# =====================================================

X_imu    = np.array(X_imu,    dtype=np.float32)
X_imu_ir = np.array(X_imu_ir, dtype=np.float32)
X_imu_radar = np.array(X_imu_radar, dtype=np.float32)
X_all    = np.array(X_all,    dtype=np.float32)
metadata = pd.DataFrame(metadata_rows)

# =====================================================
# SAVE
# =====================================================

np.save(os.path.join(OUTPUT_DIR, "X_imu_raw.npy"),           X_imu)
np.save(os.path.join(OUTPUT_DIR, "X_imu_ir_raw.npy"),        X_imu_ir)
np.save(os.path.join(OUTPUT_DIR, "X_imu_radar_raw.npy"),     X_imu_radar)
np.save(os.path.join(OUTPUT_DIR, "X_imu_ir_radar_raw.npy"),  X_all)

metadata.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

# =====================================================
# SUMMARY
# =====================================================

print("\nDone.")
print(f"X_imu_raw:          {X_imu.shape}")
print(f"X_imu_ir_raw:       {X_imu_ir.shape}")
print(f"X_imu_radar_raw:    {X_imu_radar.shape}")
print(f"X_imu_ir_radar_raw: {X_all.shape}")
print(f"metadata:           {metadata.shape}")