#!/usr/bin/env python3
"""
04_build_complete_dataset.py

Build the canonical complete real IMU + Radar dataset.

For each labeled window (from 03_label_synchronized_timeline.py):
  1. Extract the 6-second window from the synchronized segment
  2. Remove the timestamp column
  3. Resample each feature to exactly 32 frames via linear interpolation
  4. Stack into a (N, 32, 1137) array

Output:
    Complete_Dataset/
      X_imu_radar.npy          - (N, 32, 1137) float32
      metadata.csv             - Per-sample metadata
      dataset_summary.csv      - Overall statistics
      label_distribution.csv   - FALL vs NON_FALL counts
      subject_distribution.csv - Per-subject counts
      dataset_config.json      - Construction parameters

Usage:
    python 04_build_complete_dataset.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SYNC_DIR = PROJECT_DIR / "Synchronized"
LABEL_DIR = PROJECT_DIR / "Labeled_Timeline"
OUT_DIR = PROJECT_DIR / "Complete_Dataset"

TARGET_FRAMES = 32
EXPECTED_IMU_FEATURES = 3
# Radar features will be determined from actual data
WINDOW_DURATION_S = 6.0
FALL_BEFORE_S = 3.0
FALL_AFTER_S = 3.0

# Read NON_FALL_STRIDE_S from labeling config to stay in sync
_labeling_config_path = LABEL_DIR / "labeling_config.json"
if _labeling_config_path.exists():
    import json as _json
    with open(_labeling_config_path) as _f:
        _lab_cfg = _json.load(_f)
    NON_FALL_STRIDE_S = _lab_cfg.get("non_fall_stride_seconds", 3.0)
else:
    NON_FALL_STRIDE_S = 3.0  # fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_window(features: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample a (n_frames, n_features) array to (target_frames, n_features)
    using linear interpolation.
    """
    n = features.shape[0]
    n_feat = features.shape[1]

    old_time = np.linspace(0, 1, n)
    new_time = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, n_feat), dtype=np.float64)
    for feat_idx in range(n_feat):
        resampled[:, feat_idx] = np.interp(new_time, old_time, features[:, feat_idx])

    return resampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING COMPLETE REAL DATASET")
    print("=" * 60)

    # Load window definitions
    windows_path = LABEL_DIR / "windows.csv"
    if not windows_path.exists():
        print(f"ERROR: Windows file not found: {windows_path}")
        print("Run 03_label_synchronized_timeline.py first.")
        sys.exit(1)

    windows_df = pd.read_csv(windows_path)
    print(f"Windows loaded: {len(windows_df)}")
    print(f"  FALL: {len(windows_df[windows_df['label'] == 'FALL'])}")
    print(f"  NON_FALL: {len(windows_df[windows_df['label'] == 'NON_FALL'])}")

    # Load fall coverage report
    coverage_path = LABEL_DIR / "fall_coverage.csv"
    if coverage_path.exists():
        coverage_df = pd.read_csv(coverage_path)
    else:
        coverage_df = pd.DataFrame()

    # Cache: loaded segment data
    segment_cache = {}

    def get_segment_data(subject_id, segment_id):
        key = (subject_id, segment_id)
        if key not in segment_cache:
            seg_file = SYNC_DIR / f"subject_{subject_id}" / f"segment_{segment_id:03d}.csv"
            if not seg_file.exists():
                raise FileNotFoundError(f"Segment file not found: {seg_file}")
            data = np.loadtxt(seg_file, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            segment_cache[key] = data
        return segment_cache[key]

    # Determine radar feature count from the first segment
    first_row = windows_df.iloc[0]
    first_data = get_segment_data(int(first_row["subject_id"]), int(first_row["segment_id"]))
    total_cols = first_data.shape[1]
    # Columns: timestamp(1) + IMU(3) + Radar(?)
    radar_features = total_cols - 1 - EXPECTED_IMU_FEATURES
    total_features = EXPECTED_IMU_FEATURES + radar_features
    print(f"\nDetected column layout:")
    print(f"  Total columns in segment: {total_cols}")
    print(f"  Timestamp: 1")
    print(f"  IMU features: {EXPECTED_IMU_FEATURES}")
    print(f"  Radar features: {radar_features}")
    print(f"  Total features: {total_features}")

    if total_features != 1137:
        print(f"\n  *** NOTE: Total features = {total_features}, expected 1137 ***")
        print(f"  *** Proceeding with actual feature count ***")

    # Process each window
    all_samples = []
    metadata_rows = []
    errors = []

    for idx, row in windows_df.iterrows():
        sample_id = int(row["sample_id"])
        label = row["label"]
        subject_id = int(row["subject_id"])
        segment_id = int(row["segment_id"])
        w_start = float(row["window_start"])
        w_end = float(row["window_end"])
        center_time = float(row["center_time"])
        fall_number = row["fall_number"]
        fall_id = row["fall_id"]

        try:
            seg_data = get_segment_data(subject_id, segment_id)
        except FileNotFoundError as e:
            errors.append(f"Sample {sample_id}: {e}")
            continue

        seg_ts = seg_data[:, 0]
        seg_features = seg_data[:, 1:]  # (n, total_features)

        # Extract window: all frames within [w_start, w_end]
        mask = (seg_ts >= w_start) & (seg_ts <= w_end)
        window_ts = seg_ts[mask]
        window_features = seg_features[mask]

        original_frame_count = len(window_ts)

        if original_frame_count < 2:
            errors.append(
                f"Sample {sample_id} (S{subject_id}, {label}): "
                f"only {original_frame_count} frames in window"
            )
            continue

        # Verify window duration
        actual_duration = window_ts[-1] - window_ts[0]

        # Verify feature count
        if window_features.shape[1] != total_features:
            errors.append(
                f"Sample {sample_id}: feature count {window_features.shape[1]} != {total_features}"
            )
            continue

        # Resample to TARGET_FRAMES
        resampled = resample_window(window_features, TARGET_FRAMES)

        # Verify no NaN/Inf
        if np.isnan(resampled).any():
            errors.append(f"Sample {sample_id}: NaN after resampling")
            continue
        if np.isinf(resampled).any():
            errors.append(f"Sample {sample_id}: Inf after resampling")
            continue

        all_samples.append(resampled)

        metadata_rows.append({
            "sample_id": len(all_samples) - 1,  # re-index sequentially
            "label": label,
            "subject_id": subject_id,
            "fall_id": fall_id if pd.notna(fall_id) and fall_id != "" else "",
            "fall_number": int(fall_number) if pd.notna(fall_number) and fall_number != "" else "",
            "segment_id": segment_id,
            "center_time": center_time,
            "window_start": w_start,
            "window_end": w_end,
            "original_frame_count": original_frame_count,
            "target_frames": TARGET_FRAMES,
            "feature_count": total_features,
        })

    # Report errors
    if errors:
        print(f"\n*** {len(errors)} ERRORS during extraction ***")
        for e in errors:
            print(f"  {e}")

    if not all_samples:
        print("\nERROR: No valid samples extracted!")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build final arrays
    # -----------------------------------------------------------------------
    X = np.array(all_samples, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    print(f"\n{'='*60}")
    print("DATASET BUILT")
    print(f"{'='*60}")
    print(f"  X.shape = {X.shape}")
    print(f"  X.dtype = {X.dtype}")
    print(f"  NaN count: {np.isnan(X).sum()}")
    print(f"  Inf count: {np.isinf(X).sum()}")
    print(f"  Metadata rows: {len(metadata)}")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    checks = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        checks.append((name, status))
        print(f"  [{status}] {name}")

    check("X.ndim == 3", X.ndim == 3)
    check(f"X.shape[1] == {TARGET_FRAMES}", X.shape[1] == TARGET_FRAMES)
    check(f"X.shape[2] == {total_features}", X.shape[2] == total_features)
    check("X.dtype == float32", X.dtype == np.float32)
    check("No NaN in X", np.isnan(X).sum() == 0)
    check("No Inf in X", np.isinf(X).sum() == 0)
    check("len(X) == len(metadata)", len(X) == len(metadata))
    check("Labels only FALL/NON_FALL",
          set(metadata["label"].unique()) <= {"FALL", "NON_FALL"})

    # Check no duplicate windows
    window_keys = metadata[["subject_id", "window_start", "window_end", "label"]].apply(
        lambda r: f"{r['subject_id']}_{r['window_start']:.6f}_{r['window_end']:.6f}_{r['label']}", axis=1
    )
    check("No duplicate window definitions", window_keys.nunique() == len(metadata))

    # Check no NON_FALL overlaps FALL
    n_overlap_violations = 0
    for subj in metadata["subject_id"].unique():
        subj_meta = metadata[metadata["subject_id"] == subj]
        falls = subj_meta[subj_meta["label"] == "FALL"]
        non_falls = subj_meta[subj_meta["label"] == "NON_FALL"]
        for _, nf_row in non_falls.iterrows():
            for _, f_row in falls.iterrows():
                if (nf_row["window_start"] < f_row["window_end"] and
                        f_row["window_start"] < nf_row["window_end"]):
                    n_overlap_violations += 1
    check("No NON_FALL overlaps FALL", n_overlap_violations == 0)

    all_passed = all(s == "PASS" for _, s in checks)
    print(f"\n  All checks passed: {all_passed}")

    if not all_passed:
        print("\n  *** WARNING: Some validation checks FAILED ***")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    # X_imu_radar.npy
    x_path = OUT_DIR / "X_imu_radar.npy"
    np.save(x_path, X)
    print(f"  Saved: {x_path} ({X.shape})")

    # metadata.csv
    meta_path = OUT_DIR / "metadata.csv"
    metadata.to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path}")

    # label_distribution.csv
    label_dist = metadata["label"].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    label_dist_path = OUT_DIR / "label_distribution.csv"
    label_dist.to_csv(label_dist_path, index=False)
    print(f"  Saved: {label_dist_path}")

    # subject_distribution.csv
    subj_dist = metadata.groupby("subject_id")["label"].value_counts().unstack(fill_value=0)
    subj_dist = subj_dist.reset_index()
    if "FALL" not in subj_dist.columns:
        subj_dist["FALL"] = 0
    if "NON_FALL" not in subj_dist.columns:
        subj_dist["NON_FALL"] = 0
    subj_dist = subj_dist.rename(columns={"FALL": "fall_count", "NON_FALL": "non_fall_count"})
    subj_dist["total_count"] = subj_dist["fall_count"] + subj_dist["non_fall_count"]
    subj_dist_path = OUT_DIR / "subject_distribution.csv"
    subj_dist.to_csv(subj_dist_path, index=False)
    print(f"  Saved: {subj_dist_path}")

    # dataset_summary.csv
    n_covered = 0
    n_uncovered = 0
    n_annotated = 0
    if len(coverage_df) > 0:
        n_annotated = len(coverage_df)
        n_covered = int(coverage_df["covered"].sum())
        n_uncovered = n_annotated - n_covered

    summary_data = {
        "total_samples": [len(X)],
        "fall_samples": [int((metadata["label"] == "FALL").sum())],
        "non_fall_samples": [int((metadata["label"] == "NON_FALL").sum())],
        "number_of_subjects": [int(metadata["subject_id"].nunique())],
        "target_frames": [TARGET_FRAMES],
        "imu_features": [EXPECTED_IMU_FEATURES],
        "radar_features": [radar_features],
        "total_features": [total_features],
        "window_duration_seconds": [WINDOW_DURATION_S],
        "non_fall_stride_seconds": [NON_FALL_STRIDE_S],
        "nan_count": [int(np.isnan(X).sum())],
        "inf_count": [int(np.isinf(X).sum())],
        "annotated_falls": [n_annotated],
        "covered_falls": [n_covered],
        "uncovered_falls": [n_uncovered],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = OUT_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # dataset_config.json
    # Load sync config if available
    sync_config_path = SYNC_DIR / "sync_config.json"
    sync_config = {}
    if sync_config_path.exists():
        with open(sync_config_path) as f:
            sync_config = json.load(f)

    n_valid_falls = int((metadata["label"] == "FALL").sum()) if len(metadata) > 0 else 0

    dataset_config = {
        "modality": "IMU_RADAR",
        "imu_features": EXPECTED_IMU_FEATURES,
        "radar_features": radar_features,
        "total_features": total_features,
        "target_frames": TARGET_FRAMES,
        "window_duration_seconds": WINDOW_DURATION_S,
        "fall_before_seconds": FALL_BEFORE_S,
        "fall_after_seconds": FALL_AFTER_S,
        "non_fall_stride_seconds": NON_FALL_STRIDE_S,
        "radar_already_padded": True,
        "synthetic_data_included": False,
        "valid_falls": n_valid_falls,
        "excluded_annotation": "Subject 3 / Fall 2 (previously_removed_invalid_fall)",
        "dtype": "float32",
        "synchronization_config": sync_config,
    }
    config_path = OUT_DIR / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(dataset_config, f, indent=2)
    print(f"  Saved: {config_path}")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL DATASET REPORT")
    print(f"{'='*60}")
    print(f"  X_imu_radar.npy shape: {X.shape}")
    print(f"  X_imu_radar.npy dtype: {X.dtype}")
    print(f"  Total samples: {len(X)}")
    print(f"  FALL samples: {int((metadata['label'] == 'FALL').sum())}")
    print(f"  NON_FALL samples: {int((metadata['label'] == 'NON_FALL').sum())}")
    print(f"  Subjects: {sorted(metadata['subject_id'].unique().tolist())}")
    print(f"  Features: {EXPECTED_IMU_FEATURES} IMU + {radar_features} Radar = {total_features}")
    print(f"  Frames per sample: {TARGET_FRAMES}")
    print(f"  Window duration: {WINDOW_DURATION_S}s")
    print(f"  NaN: {np.isnan(X).sum()}")
    print(f"  Inf: {np.isinf(X).sum()}")
    print(f"  Annotated falls: {n_annotated}")
    print(f"  Covered falls: {n_covered}")
    print(f"  Uncovered falls: {n_uncovered}")

    print(f"\n  Feature ordering:")
    print(f"    [0]   IMU_x")
    print(f"    [1]   IMU_y")
    print(f"    [2]   IMU_z")
    print(f"    [3:{3+radar_features}] Radar_feature_1 .. Radar_feature_{radar_features}")

    print(f"\n  Per-subject:")
    for _, r in subj_dist.iterrows():
        print(f"    Subject {int(r['subject_id'])}: "
              f"{int(r['fall_count'])} FALL, {int(r['non_fall_count'])} NON_FALL, "
              f"{int(r['total_count'])} total")

    if n_uncovered > 0 and len(coverage_df) > 0:
        print(f"\n  Excluded falls:")
        uncov = coverage_df[coverage_df["covered"] == False]
        for _, r in uncov.iterrows():
            print(f"    Subject {r['subject_id']}, Fall {r['fall_number']}: {r['reason']}")

    print(f"\n{'='*60}")
    print("COMPLETE DATASET BUILD FINISHED")
    print(f"{'='*60}")

    # Quick verification load
    X_check = np.load(x_path)
    print(f"\nVerification load:")
    print(f"  np.load('{x_path.name}').shape = {X_check.shape}")
    print(f"  np.load('{x_path.name}').dtype = {X_check.dtype}")
    print(f"  NaN = {np.isnan(X_check).sum()}")
    print(f"  Inf = {np.isinf(X_check).sum()}")


if __name__ == "__main__":
    main()
