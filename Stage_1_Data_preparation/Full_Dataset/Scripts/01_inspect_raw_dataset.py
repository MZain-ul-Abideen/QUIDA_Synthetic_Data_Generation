#!/usr/bin/env python3
"""
01_inspect_raw_dataset.py

Inspect the raw QUIDA dataset to establish:
  - IMU file structure (shape, columns, sampling frequency, timestamps, NaN/Inf)
  - Radar file structure (shape, columns, sampling frequency, timestamps, NaN/Inf)
  - Falls annotation file structure (format, subjects, fall counts)

This script does NOT modify any data.
It produces a human-readable inspection report.

Usage:
    python 01_inspect_raw_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "Raw_data"

SUBJECTS = list(range(1, 11))  # 1..10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inspect_sensor_file(filepath: Path, sensor_name: str) -> dict:
    """Inspect a single sensor CSV file and return a summary dict."""
    info = {"file": filepath.name, "sensor": sensor_name, "exists": filepath.exists()}
    if not filepath.exists():
        return info

    df = pd.read_csv(filepath, header=None)
    info["shape"] = df.shape
    info["n_rows"] = df.shape[0]
    info["n_cols"] = df.shape[1]
    info["dtypes"] = df.dtypes.value_counts().to_dict()

    # Timestamp analysis (column 0)
    ts = df.iloc[:, 0].values.astype(np.float64)
    info["ts_min"] = float(ts.min())
    info["ts_max"] = float(ts.max())
    info["ts_duration_s"] = float(ts.max() - ts.min())

    diffs = np.diff(ts)
    info["ts_median_interval_s"] = float(np.median(diffs))
    info["ts_mean_interval_s"] = float(np.mean(diffs))
    info["ts_std_interval_s"] = float(np.std(diffs))
    info["estimated_freq_hz"] = float(1.0 / np.median(diffs)) if np.median(diffs) > 0 else 0.0

    # Monotonicity
    info["ts_strictly_increasing"] = bool(np.all(diffs > 0))
    info["ts_non_decreasing"] = bool(np.all(diffs >= 0))
    n_dup = int(np.sum(diffs == 0))
    info["ts_duplicate_count"] = n_dup
    n_negative = int(np.sum(diffs < 0))
    info["ts_negative_diff_count"] = n_negative

    # Feature columns (everything after timestamp)
    features = df.iloc[:, 1:].values.astype(np.float64)
    info["feature_cols"] = df.shape[1] - 1
    info["nan_count"] = int(np.isnan(features).sum())
    info["inf_count"] = int(np.isinf(features).sum())
    info["feature_min"] = float(np.nanmin(features))
    info["feature_max"] = float(np.nanmax(features))
    info["feature_mean"] = float(np.nanmean(features))

    # Large gaps (>5× median interval)
    median_dt = np.median(diffs)
    if median_dt > 0:
        gap_threshold = 5.0 * median_dt
        large_gaps = np.where(diffs > gap_threshold)[0]
        info["large_gap_count"] = int(len(large_gaps))
        if len(large_gaps) > 0:
            info["large_gaps"] = [
                {
                    "index": int(idx),
                    "gap_seconds": float(diffs[idx]),
                    "ts_before": float(ts[idx]),
                    "ts_after": float(ts[idx + 1]),
                }
                for idx in large_gaps[:20]  # cap at 20 for readability
            ]
        else:
            info["large_gaps"] = []
    else:
        info["large_gap_count"] = -1
        info["large_gaps"] = []

    return info


def inspect_falls(filepath: Path) -> dict:
    """Inspect the Falls.csv annotation file."""
    info = {"file": filepath.name, "exists": filepath.exists()}
    if not filepath.exists():
        return info

    df = pd.read_csv(filepath, header=None)
    info["shape"] = df.shape
    info["n_rows"] = df.shape[0]
    info["n_cols"] = df.shape[1]

    # Each column = one subject, each row = one fall
    # Values should be Unix timestamps
    info["subjects_inferred"] = df.shape[1]
    info["falls_per_subject"] = df.shape[0]
    info["total_annotations"] = int(df.notna().sum().sum())

    # Check for NaN entries
    nan_locations = []
    for col_idx in range(df.shape[1]):
        for row_idx in range(df.shape[0]):
            if pd.isna(df.iloc[row_idx, col_idx]):
                nan_locations.append(
                    {"subject_col": col_idx, "fall_row": row_idx}
                )
    info["nan_annotations"] = nan_locations
    info["nan_count"] = len(nan_locations)

    # Print all fall timestamps per subject
    subject_falls = {}
    for col_idx in range(df.shape[1]):
        subj_id = col_idx + 1
        vals = df.iloc[:, col_idx].dropna().values.astype(np.float64)
        subject_falls[subj_id] = vals.tolist()
    info["subject_falls"] = subject_falls

    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("QUIDA RAW DATASET INSPECTION")
    print("=" * 80)
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Directory exists: {RAW_DIR.exists()}")
    print()

    if not RAW_DIR.exists():
        print("ERROR: Raw_data directory does not exist!")
        sys.exit(1)

    # List all files
    all_files = sorted(RAW_DIR.iterdir())
    print(f"Files found ({len(all_files)}):")
    for f in all_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s}  {size_kb:10.1f} KB")
    print()

    # -----------------------------------------------------------------------
    # Inspect Falls
    # -----------------------------------------------------------------------
    falls_path = RAW_DIR / "falls.csv"
    if not falls_path.exists():
        # Try alternative capitalizations
        for name in ["Falls.csv", "FALLS.csv", "falls.CSV"]:
            alt = RAW_DIR / name
            if alt.exists():
                falls_path = alt
                break

    print("-" * 80)
    print("FALLS ANNOTATION FILE")
    print("-" * 80)
    falls_info = inspect_falls(falls_path)
    if not falls_info["exists"]:
        print(f"  WARNING: Falls file not found at {falls_path}")
    else:
        print(f"  File: {falls_info['file']}")
        print(f"  Shape: {falls_info['shape']}")
        print(f"  Subjects (columns): {falls_info['subjects_inferred']}")
        print(f"  Falls per subject (rows): {falls_info['falls_per_subject']}")
        print(f"  Total non-NaN annotations: {falls_info['total_annotations']}")
        print(f"  NaN annotations: {falls_info['nan_count']}")
        if falls_info["nan_count"] > 0:
            print("  NaN locations:")
            for loc in falls_info["nan_annotations"]:
                print(f"    subject_col={loc['subject_col']} (subject {loc['subject_col']+1}), "
                      f"fall_row={loc['fall_row']} (fall {loc['fall_row']+1})")

        print("\n  Fall timestamps per subject:")
        for subj_id, timestamps in falls_info["subject_falls"].items():
            print(f"    Subject {subj_id}: {len(timestamps)} falls")
            for i, ts in enumerate(timestamps):
                print(f"      Fall {i+1}: {ts:.6f}")
    print()

    # -----------------------------------------------------------------------
    # Inspect IMU and Radar for each subject
    # -----------------------------------------------------------------------
    imu_summaries = []
    radar_summaries = []

    for subj in SUBJECTS:
        imu_path = RAW_DIR / f"subject_{subj}_accelerometer.csv"
        radar_path = RAW_DIR / f"subject_{subj}_radar.csv"

        print("-" * 80)
        print(f"SUBJECT {subj}")
        print("-" * 80)

        # --- IMU ---
        print(f"\n  IMU (accelerometer):")
        imu_info = inspect_sensor_file(imu_path, "IMU")
        if not imu_info["exists"]:
            print(f"    WARNING: File not found: {imu_path.name}")
        else:
            print(f"    File: {imu_info['file']}")
            print(f"    Shape: {imu_info['shape']}")
            print(f"    Columns: {imu_info['n_cols']} (1 timestamp + {imu_info['feature_cols']} features)")
            print(f"    Timestamp range: {imu_info['ts_min']:.6f} -> {imu_info['ts_max']:.6f}")
            print(f"    Duration: {imu_info['ts_duration_s']:.2f} s")
            print(f"    Median interval: {imu_info['ts_median_interval_s']*1000:.3f} ms")
            print(f"    Estimated frequency: {imu_info['estimated_freq_hz']:.2f} Hz")
            print(f"    Strictly increasing: {imu_info['ts_strictly_increasing']}")
            print(f"    Duplicate timestamps: {imu_info['ts_duplicate_count']}")
            print(f"    Negative diffs: {imu_info['ts_negative_diff_count']}")
            print(f"    NaN count: {imu_info['nan_count']}")
            print(f"    Inf count: {imu_info['inf_count']}")
            print(f"    Feature range: [{imu_info['feature_min']:.4f}, {imu_info['feature_max']:.4f}]")
            print(f"    Large gaps (>5× median): {imu_info['large_gap_count']}")
            if imu_info["large_gaps"]:
                for g in imu_info["large_gaps"][:5]:
                    print(f"      index {g['index']}: gap={g['gap_seconds']:.4f}s")
        imu_summaries.append(imu_info)

        # --- Radar ---
        print(f"\n  RADAR:")
        radar_info = inspect_sensor_file(radar_path, "Radar")
        if not radar_info["exists"]:
            print(f"    WARNING: File not found: {radar_path.name}")
        else:
            print(f"    File: {radar_info['file']}")
            print(f"    Shape: {radar_info['shape']}")
            print(f"    Columns: {radar_info['n_cols']} (1 timestamp + {radar_info['feature_cols']} features)")
            print(f"    Timestamp range: {radar_info['ts_min']:.6f} -> {radar_info['ts_max']:.6f}")
            print(f"    Duration: {radar_info['ts_duration_s']:.2f} s")
            print(f"    Median interval: {radar_info['ts_median_interval_s']*1000:.3f} ms")
            print(f"    Estimated frequency: {radar_info['estimated_freq_hz']:.2f} Hz")
            print(f"    Strictly increasing: {radar_info['ts_strictly_increasing']}")
            print(f"    Duplicate timestamps: {radar_info['ts_duplicate_count']}")
            print(f"    Negative diffs: {radar_info['ts_negative_diff_count']}")
            print(f"    NaN count: {radar_info['nan_count']}")
            print(f"    Inf count: {radar_info['inf_count']}")
            print(f"    Feature range: [{radar_info['feature_min']:.6f}, {radar_info['feature_max']:.6f}]")
            print(f"    Large gaps (>5× median): {radar_info['large_gap_count']}")
            if radar_info["large_gaps"]:
                for g in radar_info["large_gaps"][:5]:
                    print(f"      index {g['index']}: gap={g['gap_seconds']:.4f}s")

            # Verify expected 1135 columns
            if radar_info["n_cols"] != 1135:
                print(f"\n    *** UNEXPECTED: Radar has {radar_info['n_cols']} columns, expected 1135 ***")
                print(f"    *** This means {radar_info['feature_cols']} features instead of expected 1134 ***")

        radar_summaries.append(radar_info)

        # --- Temporal overlap ---
        if imu_info["exists"] and radar_info["exists"]:
            overlap_start = max(imu_info["ts_min"], radar_info["ts_min"])
            overlap_end = min(imu_info["ts_max"], radar_info["ts_max"])
            overlap_dur = max(0, overlap_end - overlap_start)
            print(f"\n  Temporal overlap:")
            print(f"    IMU  : [{imu_info['ts_min']:.2f}, {imu_info['ts_max']:.2f}]")
            print(f"    Radar: [{radar_info['ts_min']:.2f}, {radar_info['ts_max']:.2f}]")
            print(f"    Overlap: {overlap_dur:.2f} s")
            if overlap_dur <= 0:
                print(f"    *** WARNING: No temporal overlap! ***")
        print()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Subj':>5s} | {'IMU rows':>10s} | {'IMU cols':>9s} | {'IMU Hz':>8s} | {'Radar rows':>11s} | {'Radar cols':>10s} | {'Radar Hz':>9s} | {'Overlap(s)':>10s}"
    print(header)
    print("-" * len(header))
    for i, subj in enumerate(SUBJECTS):
        imu = imu_summaries[i]
        rad = radar_summaries[i]
        if imu["exists"] and rad["exists"]:
            overlap = max(0, min(imu["ts_max"], rad["ts_max"]) - max(imu["ts_min"], rad["ts_min"]))
            print(f"{subj:5d} | {imu['n_rows']:10d} | {imu['n_cols']:9d} | {imu['estimated_freq_hz']:8.2f} | {rad['n_rows']:11d} | {rad['n_cols']:10d} | {rad['estimated_freq_hz']:9.2f} | {overlap:10.2f}")
        else:
            print(f"{subj:5d} | {'MISSING':>10s} | {'':>9s} | {'':>8s} | {'MISSING':>11s} | {'':>10s} | {'':>9s} | {'':>10s}")

    # -----------------------------------------------------------------------
    # Radar column consistency check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RADAR COLUMN CONSISTENCY")
    print("=" * 80)
    col_counts = set()
    for i, subj in enumerate(SUBJECTS):
        rad = radar_summaries[i]
        if rad["exists"]:
            col_counts.add(rad["n_cols"])
            print(f"  Subject {subj}: {rad['n_cols']} columns ({rad['feature_cols']} features)")
    if len(col_counts) == 1:
        print(f"\n  All radar files have consistent column count: {col_counts.pop()}")
    else:
        print(f"\n  *** WARNING: Inconsistent radar column counts: {col_counts} ***")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
